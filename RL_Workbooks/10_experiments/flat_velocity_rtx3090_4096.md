# Trying flat velocity 4096 on RTX 3090

## Training

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 1024 \
  --agent.max_iterations 5000 \
  --agent.run-name experiment-flat-baseline \
  --agent.logger tensorboard
```

## Initial output

```
################################################################################
                           Learning iteration 4/5000                             

                               Run name: experiment-flat-baseline
                            Total steps: 122880 
                       Steps per second: 10173 
                        Collection time: 2.247s 
                          Learning time: 0.169s 
                        Mean value loss: 0.0485
                    Mean surrogate loss: -0.0081
                      Mean entropy loss: 19.6862
                            Mean reward: 0.31
                    Mean episode length: 35.02
                        Mean action std: 0.99
   Episode_Reward/track_linear_velocity: 0.0258
  Episode_Reward/track_angular_velocity: 0.0019
                 Episode_Reward/upright: 0.0275
                    Episode_Reward/pose: 0.0083
            Episode_Reward/body_ang_vel: -0.0238
        Episode_Reward/angular_momentum: -0.0000
          Episode_Reward/dof_pos_limits: -0.0006
          Episode_Reward/action_rate_l2: -0.0932
                Episode_Reward/air_time: 0.0123
          Episode_Reward/foot_clearance: -0.0004
       Episode_Reward/foot_swing_height: -0.0018
               Episode_Reward/foot_slip: -0.0001
         Episode_Reward/self_collisions: -0.0005
      Episode_Reward/head_pose_tracking: 0.0582
      Episode_Reward/body_pose_tracking: 0.0000
          Episode_Reward/head_pose_bias: 0.0000
        Episode_Metrics/mean_action_acc: 1.8715
          Curriculum/action_rate_weight: -0.1000
               Curriculum/standing_envs: 0.0200
             Curriculum/head_pose_range: 0.0700
             Curriculum/body_pose_range: 0.0500
                   Curriculum/com_range: 0.0030
              Curriculum/head_com_range: 0.0030
       Curriculum/head_pose_bias_weight: 0.0000
             Metrics/twist/error_vel_xy: 0.0302
            Metrics/twist/error_vel_yaw: 0.1760
           Episode_Termination/time_out: 0.0000
          Episode_Termination/fell_over: 30.3333
Episode_Termination/out_of_terrain_bounds: 0.0000
          Episode_Termination/nan_state: 0.0000
          Metrics/angular_momentum_mean: 0.0115
                  Metrics/air_time_mean: 0.0605
               Metrics/peak_height_mean: 0.0110
             Metrics/slip_velocity_mean: 0.1028
--------------------------------------------------------------------------------
                         Iteration time: 2.42s
                           Time elapsed: 00:00:12
                                    ETA: 03:34:54
```

## nvidia-smi

```
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090 Ti     On  |   00000000:01:00.0  On |                  Off |
| 52%   54C    P2            132W /  450W |    3306MiB /  24564MiB |     34%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

--------------------------------------------------------------------------------
                         Iteration time: 1.89s
                           Time elapsed: 02:36:54
                                    ETA: 00:00

## Watch microduck go

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "logs/rsl_rl/velocity/2026-09-17_07-00-22_experiment-flat-baseline/model_4999.pt" \
  --num-envs 1 \
  --viewer viser
```

## Outcome

microduck followed arrow but fell over a lot