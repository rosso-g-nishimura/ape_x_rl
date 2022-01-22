import datetime
import numpy as np
import os
from psutil import cpu_count
import ray

from learner import Learner
from actor import Actor
from replay import Experiences
from logger import XLogger


if __name__ == '__main__':
    actor_count = cpu_count() - 2
    ray.init(num_gpus=1, log_to_driver=False)
    # ray.init(num_gpus=1, log_to_driver=False, local_mode=True)
    print(ray.cluster_resources())

    # 保存先作成
    save_dir = os.getcwd()
    save_dir = os.path.join(save_dir, 'checkpoints',
                            datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.mkdir(save_dir)
    logger = XLogger(save_dir)

    # # 学習済み
    # load_dir = os.path.join(os.getcwd(), 'checkpoints', '20211010153713')
    # load_weight = 'sync_000884.chkpt'

    try:
        # この辺のパラメータを、ログとしてテキスト出力したい（mainにパラメータを集約したい）
        state_dim = (4, 84, 84)
        action_dim = 5  # 7
        batch_size = 32
        sample_size = 16
        n_step = 3
        gamma = 0.99
        target_sync = 500
        ready_learn = 20

        actor_steps = 1_000
        actor_loop_cnt = 100_000
        done_avg_x = 9_000

        # replay
        replay = Experiences(300_000, 0.99, 3)

        def get_sample_batch():
            btc, idx = [], []
            for _ in range(sample_size):
                replay_sample = replay.sample(batch_size)
                btc.append(replay_sample[0])
                idx.append(replay_sample[1])
            return btc, idx

        # learner

        learn_cnt = 0
        sync_cnt = 0
        pre_sync_cnt = sync_cnt
        learner = Learner.remote(state_dim, action_dim,
                                 batch_size, n_step, gamma, target_sync, save_dir)

        # learn_cnt = 27625
        # sync_cnt = 884
        # pre_sync_cnt = sync_cnt
        # learner = Learner.remote(state_dim, action_dim,
        #                          batch_size, n_step, gamma, target_sync, save_dir,
        #                          load_path=os.path.join(load_dir, load_weight), sync_cnt=sync_cnt*target_sync)

        weight_path = ray.get(learner.get_weight_path.remote())
        weight_path = ray.put(weight_path)

        # actor
        eps = np.linspace(0.01, 0.5, actor_count)
        if actor_count == 1:
            eps = [0.3]
        actor_list = [Actor.remote(i, state_dim, weight_path,
                                   steps=actor_steps, epsilon=eps[i]) for i in range(actor_count)]
        running_actor = [actor.run.remote(weight_path) for actor in actor_list]
        # tester
        test_actor = Actor.remote(actor_count, state_dim, weight_path, test_mode=True)
        testing_actor = test_actor.test.remote(weight_path)
        recent_test_x = []

        # 一定量replayにデータを貯める
        for cnt in range(ready_learn):
            actor_data, running_actor = ray.wait(running_actor, num_returns=1)
            batch, actor_priority, actor_id = ray.get(actor_data[0])
            replay.put(batch, actor_priority)
            running_actor.extend([actor_list[actor_id].run.remote(weight_path)])

        # 本番開始
        sample_batch, _ = get_sample_batch()
        learning_learner = learner.learn.remote(sample_batch)
        sample_batch, index = get_sample_batch()

        # training
        for cnt in range(actor_loop_cnt):
            actor_data, running_actor = ray.wait(running_actor, num_returns=1)
            batch, actor_priority, actor_id = ray.get(actor_data[0])
            replay.put(batch, actor_priority)
            running_actor.extend([actor_list[actor_id].run.remote(weight_path)])

            # learn
            learner_data, _ = ray.wait([learning_learner], timeout=0)
            if learner_data:
                learner_priority, weight_path, sync_cnt, loss, Q = ray.get(learner_data[0])
                weight_path = ray.put(weight_path)
                learning_learner = learner.learn.remote(sample_batch)
                replay.update(index, learner_priority)
                sample_batch, index = get_sample_batch()
                learn_cnt += 1
                print('learner updated:', learn_cnt)

                # test
                if sync_cnt > pre_sync_cnt:
                    x, reward = ray.get(testing_actor)
                    # logger.plot(x, pre_sync_cnt)
                    logger.plot(x, loss, Q, reward, pre_sync_cnt)
                    pre_sync_cnt = sync_cnt
                    testing_actor = test_actor.test.remote(weight_path)

                    # 過去10回のテスト平均が規定値をこえたら終了
                    recent_test_x.append(x)
                    if len(recent_test_x) > 10:
                        recent_test_x = recent_test_x[-10:]
                    avg_x = sum(recent_test_x) / len(recent_test_x)
                    print('recent average is ', avg_x)
                    if avg_x > done_avg_x:
                        print('recent average is over', done_avg_x)
                        break

    except Exception as e:
        print(e)
    finally:
        ray.shutdown()
