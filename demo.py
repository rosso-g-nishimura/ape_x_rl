from matplotlib import animation
import matplotlib.pyplot as plt
import os
import torch

from demo_actor import DemoActor


if __name__ == '__main__':
    target_dir = '20211012004616'
    target_weight = 'sync_000406'

    target_path = os.path.join(os.getcwd(), 'checkpoints')
    target_path = os.path.join(target_path, target_dir)
    target_path = os.path.join(target_path, f'{target_weight}.chkpt')
    weight = torch.load(target_path)

    # state_dim = 4
    state_dim = (4, 84, 84)
    demo_actor = DemoActor(state_dim, weight)
    frames = demo_actor.demo(video=True)

    if frames != None:
        plt.figure(figsize=(frames[0].shape[1]/72.0,
                            frames[0].shape[0]/72.0), dpi=72)
        patch = plt.imshow(frames[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(frames[i])

        anim = animation.FuncAnimation(plt.gcf(), animate,
                                       frames=len(frames), interval=50)

        target_path = os.path.join(os.getcwd(), 'checkpoints')
        target_path = os.path.join(target_path, target_dir)
        target_path = os.path.join(target_path, f'{target_weight}.mp4')
        anim.save(target_path)
