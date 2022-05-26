"""Point Maze."""
"""reference: https://github.com/FangchenLiu/map_planner/blob/master/goal_env/mujoco/point.py"""

import numpy as np
from transforms3d.quaternions import axangle2quat
from gym import spaces

import sapien.core as sapien
from sapien.core import Pose
from sapien.utils.viewer import Viewer
from sapien_env import SapienEnv

#YOU ARE NOT ALLOWED TO MODIFY THIS FILE EXCEPT FOR THE REWARD PART.

def create_point(
        scene: sapien.Scene,
        joint_friction=0.0,
        joint_damping=0.0,
        density=1000.0,
) -> sapien.Articulation:
    builder: sapien.ArticulationBuilder = scene.create_articulation_builder()

    base: sapien.LinkBuilder = builder.create_link_builder()  
    base.set_name('base')

    dummy1 = builder.create_link_builder(base)
    dummy1.set_name('dummy1')
    dummy1.set_joint_name('dummy1_joint')

    dummy1.set_joint_properties(
        'prismatic',
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        pose_in_child=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        friction=joint_friction,
        damping=joint_damping)

    dummy2 = builder.create_link_builder(dummy1)
    dummy2.set_name('dummy2')
    dummy2.set_joint_name('dummy2_joint')

    dummy2.set_joint_properties(
        'prismatic',
        limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose(p=[0, 0, 0], q=axangle2quat([0, 0, 1], np.deg2rad(90))),
        pose_in_child=sapien.Pose(p=[0, 0, 0], q=axangle2quat([0, 0, 1], np.deg2rad(90))),
        friction=joint_friction,
        damping=joint_damping,
    )

    body = builder.create_link_builder(dummy2)
    body.set_name('body')
    body.set_joint_name('body_joint')
    body.add_sphere_collision(radius = 0.2)
    body.add_sphere_visual(radius = 0.2, color=[0, 0.246, 0.566])

    body.set_joint_properties(
        "revolute",
        limits=[[-np.inf, np.inf]],  
        pose_in_parent=sapien.Pose(
            p=[0, 0, 0], q=axangle2quat([0, 1, 0], np.deg2rad(90))),
        pose_in_child=sapien.Pose(
            p=[0, 0, 0], q=axangle2quat([0, 1, 0], np.deg2rad(90))),
        friction=joint_friction,
        damping=joint_damping,
    )

    arrow = builder.create_link_builder(body)
    arrow.set_name('arrow')
    arrow.set_joint_name('arrow_joint')
    arrow.add_capsule_collision(radius=0.04, half_length=0.12)
    arrow.add_capsule_visual(radius=0.04, half_length=0.12, color=[0.363, 0.660, 0.910])

    arrow.set_joint_properties(
        "fixed",
        limits=[],  
        pose_in_parent=sapien.Pose(
            p=[0.2, 0, 0], q=[1, 0, 0, 0]),
        pose_in_child=sapien.Pose(
            p=[-0.12, 0, 0], q=[1, 0, 0, 0]),
        friction=joint_friction,
        damping=joint_damping,
    )

    point = builder.build(fix_root_link=True)
    point.set_name('point')
    point.set_pose(sapien.Pose(p=[0, 0, 0.2], q=[1, 0, 0, 0]))
    return point

def create_maze(scene: sapien.Scene, map):
    l = 0.6
    n = len(map)
    m = len(map[0])
    for i in range(n):
        for j in range(m):
            if map[n - 1 - i][m - 1 - j] != 0:
                continue
            builder: sapien.ActorBuilder = scene.create_actor_builder()
            builder.add_box_collision(half_size=[l, l, l/2], density = 1e15)
            builder.add_box_visual(half_size=[l, l, l/2], color=[1, 0.482, 0.328])
            box = builder.build_static("grid_%d_%d" % (i, j))
            box.set_pose(sapien.Pose(p=[i * l * 2 - (n - 1) * l, j * l * 2 - (m - 1) * l, l/2], q=[1, 0, 0, 0]))
    return 

class PointMazeEnv(SapienEnv):
    def __init__(self):
        self.map = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [0,1,1,1,0,0,1,1,1,0,0,1,1,1,0],
                    [0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],
                    [0,1,0,0,0,0,1,1,1,0,0,1,1,1,0],
                    [0,1,0,0,0,0,0,0,1,0,0,1,0,0,0],
                    [0,1,1,1,0,0,1,1,1,0,0,1,1,1,0],
                    [0,1,1,1,0,1,1,1,0,1,0,1,1,1,0],
                    [0,0,0,1,0,1,0,1,0,1,0,1,0,1,0],
                    [0,1,1,1,0,1,1,1,0,1,0,1,0,1,0],
                    [0,1,0,0,0,0,0,1,0,1,0,1,0,1,0],
                    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        numpymap = np.array(self.map)
        traveled = np.zeros_like(numpymap)
        traveled[5,7] = 1
        traveled[6,7] = 1
        l = []
        r = np.zeros_like(numpymap)
        r_dir = np.zeros((numpymap.shape[0], numpymap.shape[1],2))
        r[5,7] = 100
        r[6,7] = 100
        l.append((5,7))
        l.append((6,7))
        while len(l)>0:
            curx, cury = l.pop(0)
            for (x,y) in [(curx-1, cury), (curx+1, cury), (curx, cury-1), (curx, cury+1)]:
                if x< 0 or x > 11:
                    continue
                if y<0 or y>14:
                    continue
                if traveled[x,y] == 1:
                    continue
                if numpymap[x,y] < 0:
                    r[x,y] = -10
                else:
                    traveled[x,y] = 1
                    l.append((x,y))
                    curr = r[curx, cury]
                    rew = curr - 3
                    r[x,y] = rew
                    r_dir[x,y,:] = [x-curx, y-cury]
        r += (r==0) * -10
        self.rmap = r_dir
        self.rmap_static = r
        self.lastval = 0

        super().__init__(control_freq=1, timestep=0.005)
        self.n = len(self.map)
        self.m = len(self.map[0])
        self.l = 0.6
        self.maze_half_size = [(self.n - 2) * self.l, (self.m - 2) * self.l]

        self.point = self.get_articulation('point')

        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=[6],
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1,
                                       high=1,
                                       shape=[2],
                                       dtype=np.float32)

    # ---------------------------------------------------------------------------- #
    # Simulation world
    # ---------------------------------------------------------------------------- #
    def _build_world(self):
        # frictionless
        phy_mtl = self._scene.create_physical_material(0.0, 0.0, 0.0)
        self._scene.default_physical_material = phy_mtl
        create_maze(self._scene, self.map)
        create_point(self._scene)
        self._scene.add_ground(altitude=0)

    # ---------------------------------------------------------------------------- #
    # RL
    # ---------------------------------------------------------------------------- #
    def step(self, action):
        action = np.copy(action)
        action = np.clip(action, -1, 1)
        action[0] = 0.1 * action[0]

        qpos = np.copy(self.point.get_qpos())
        qpos[2] += action[1]
        ori = qpos[2]
        dx = np.cos(ori) * action[0]
        dy = -np.sin(ori) * action[0]

        qpos[0] = np.clip(qpos[0] + dx, -self.maze_half_size[0], self.maze_half_size[0])
        qpos[1] = np.clip(qpos[1] + dy, -self.maze_half_size[1], self.maze_half_size[1])


        self.point.set_qpos(qpos)
        for _ in range(self.control_freq):
            self._scene.step()

        obs = self._get_obs()
        
        done = np.sqrt((obs[0] ** 2 + obs[1] ** 2)) < 0.5
        reward = 0
        if(np.abs(obs[0])<1.2 and np.abs(obs[1])<0.6):
            curval = -np.sqrt((obs[0] ** 2 + obs[1] ** 2))
            if(self.lastval>50):
                self.lastval = 0
        else:
            l = 1.2
            def get_cur_xy(curqpos):
                # print(curqpos)
                x = int(np.floor(-curqpos[0]/l + 6))
                y = int(np.floor(-curqpos[1]/l + 7.5))
                return x, y
            
            def get_cur_loc(x,y):
                xpos = -(x-6)*l+0.3
                ypos = -(y-7.5)*l+0.3
                return xpos, ypos
            
            x,y = get_cur_xy(obs[0:2])
            curval = self.rmap_static[x,y]
        if curval < self.lastval:
            reward = -10
        elif curval > self.lastval:
            reward = 3
        self.lastval = curval
            # correct_dir = self.rmap[x,y]
            # tx = x + correct_dir[0]
            # ty = y + correct_dir[1]
            # xpos, ypos = get_cur_loc(tx, ty)
            # reward = (xpos-qpos[0])*obs[3] + (ypos-qpos[1])*obs[4] #TODO
            # print(x,y, xpos, ypos, obs, reward)

        return obs, reward, done, {}

    def reset(self, x = None, y = None, theta = None):
        if not x:
            while True:
                x = np.random.randint(self.n)
                y = np.random.randint(self.m)
                if self.map[x][y] == 1:
                    break
        x = (self.n - 2 * x - 1) * self.l
        y = (self.m - 2 * y - 1) * self.l
        if not theta:
            theta = np.random.randn()
        vel = np.zeros(3)
        self.point.set_qpos([x, y, theta])
        self.point.set_qvel(vel)
        self._scene.step()
        return self._get_obs()

    def _get_obs(self):
        qpos = self.point.get_qpos()
        qvel = self.point.get_qvel()
        return np.hstack([qpos, qvel])

    # ---------------------------------------------------------------------------- #
    # Visualization
    # ---------------------------------------------------------------------------- #
    def _setup_lighting(self):
        rscene = self._scene #.get_renderer_scene()
        rscene.set_ambient_light([0.5, 0.5, 0.5])
        rscene.add_directional_light([0, 0, -1], [1, 1, 1], shadow=True)

    def _setup_viewer(self):
        self._setup_lighting()
        self.viewer = Viewer(self._renderer)
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(0, 0, 20)
        self.viewer.set_camera_rpy(0, -1.57, 0)
        self.viewer.window.set_camera_parameters(near=0.01, far=100, fovy=1)
        #self.viewer.focus_actor(self.cart)

