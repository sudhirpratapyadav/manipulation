Now I have the full picture — let me write a thorough breakdown.
                                                                                            
  SplatSim — Deep Dive                                                                      
                                               
  ▎ Paper: arXiv 2409.10161 (ICRA 2025 Spotlight)                                           
  ▎ Code: https://github.com/qureshinomaan/SplatSim                   
  ▎ Authors: M. Nomaan Qureshi, Sparsh Garg, Francisco Yandun, David Held, George Kantor,   
  ▎ Abhisesh Silwal — CMU Robotics Institute                                                
                                                                                            
  The pitch in one line: swap mesh rendering inside PyBullet for 3D Gaussian Splat          
  rendering, train a Diffusion Policy on the photoreal sim images, deploy zero-shot to a 
  real UR5. No real-world demos needed.                                                     
                                                                      
  ---                                          
  1. The full stack
                                                                                            
  ┌──────────────┬──────────────────────────────────────────────────────────────────────┐ 
  │    Layer     │                            What they use                             │   
  ├──────────────┼──────────────────────────────────────────────────────────────────────┤ 
  │ Physics      │ PyBullet (only). Forward kinematics from PyBullet's API              │ 
  ├──────────────┼──────────────────────────────────────────────────────────────────────┤ 
  │ Rendering    │ Original 3DGS (Kerbl et al. SIGGRAPH 2023) wrapped as                │   
  │              │ gaussian-splatting-wrapper submodule                                 │   
  ├──────────────┼──────────────────────────────────────────────────────────────────────┤   
  │ Robot arm    │ UR5 + Robotiq 2F-85 gripper (single platform)                        │   
  ├──────────────┼──────────────────────────────────────────────────────────────────────┤   
  │ Real cameras │ 2× Intel RealSense D455                                              │
  ├──────────────┼──────────────────────────────────────────────────────────────────────┤   
  │ Capture      │ iPhone (for the initial scene scan that becomes the splat)           │
  │ device       │                                                                      │
  ├──────────────┼──────────────────────────────────────────────────────────────────────┤   
  │ Teleop       │ GELLO (low-cost arm-shaped teleoperator) — optional                  │
  ├──────────────┼──────────────────────────────────────────────────────────────────────┤   
  │ Policy       │ Diffusion Policy (only) — Chi et al. RSS 2023                        │
  ├──────────────┼──────────────────────────────────────────────────────────────────────┤   
  │ Deployment   │ RTX 3080 Ti (consumer-grade)                                         │
  │ GPU          │                                                                      │   
  └──────────────┴──────────────────────────────────────────────────────────────────────┘
                                               
  The simulator runs PyBullet for collision/dynamics, then for every frame they:            
  1. Take the simulator's joint state.
  2. Run forward kinematics → world transform for each robot link.                          
  3. Look up that link's pre-segmented Gaussian sub-cloud, transform it.
  4. Composite the transformed robot Gaussians + static background scene Gaussians +        
  manipuland Gaussians.                                                                     
  5. Hand the merged Gaussian set to the standard 3DGS rasterizer with a virtual camera     
  matching the real D455.                                                                   
  6. Render → that's the policy's input image.                                              
                                                                      
  ---                                                                                       
  2. The hard part: aligning the Gaussian scene to the simulator      
                                                                                            
  This is what makes SplatSim non-trivial. They don't just "render Gaussians"; they
  decompose the scene into independently controllable sub-clouds:                           
                                                                      
  ┌──────────────────────────┬──────────────────────────────────────────────────────────┐   
  │       Object class       │                  Segmentation approach                   │
  ├──────────────────────────┼──────────────────────────────────────────────────────────┤
  │ Background               │ The leftover Gaussians after subtracting                 │
  │                          │ robot/manipulands                                        │
  ├──────────────────────────┼──────────────────────────────────────────────────────────┤   
  │                          │ Manual point-cloud subtraction → ICP alignment to the    │   
  │ Robot links (per-link!)  │ URDF mesh → per-link CAD bounding boxes carve up the     │   
  │                          │ cloud                                                    │   
  ├──────────────────────────┼──────────────────────────────────────────────────────────┤   
  │ Robot gripper            │ KNN classifier trained on labeled simulator point clouds │
  │ (articulated)            │  — assigns each Gaussian to the correct moving finger    │   
  ├──────────────────────────┼──────────────────────────────────────────────────────────┤   
  │ Manipuland (apple,       │ Captured with a separate iPhone scan, segmented, treated │
  │ orange, T-block, peg)    │  as a rigid body in PyBullet                             │   
  └──────────────────────────┴──────────────────────────────────────────────────────────┘
                                               
  Then a fixed T_robot_to_splat rigid transform glues the Gaussian world to the PyBullet    
  world. Every step at runtime:
                                                                                            
  final_pose_link = (T_robot_to_splat)⁻¹  ·  T_fk_link  ·  T_robot_to_splat
                                                                                            
  is applied to the link's Gaussians.                                                       
                                                                                            
  ---                                                                                       
  3. Tasks and results                                                
                                               
  All 4 tasks done with Diffusion Policy, 40 trials each:
                                                                                            
  ┌────────────────────────┬───────────────────┬─────────────────────────────┬─────────┐    
  │          Task          │ Sim2Real (theirs) │ Real2Real (oracle baseline) │ Sim2Sim │    
  ├────────────────────────┼───────────────────┼─────────────────────────────┼─────────┤    
  │ T-Push                 │ 90% (36/40)       │ 100% (40/40)                │ 100%    │
  ├────────────────────────┼───────────────────┼─────────────────────────────┼─────────┤
  │ Pick-Up-Apple          │ 95% (38/40)       │ 100% (40/40)                │ 100%    │    
  ├────────────────────────┼───────────────────┼─────────────────────────────┼─────────┤
  │ Orange-On-Plate        │ 90% (36/40)       │ 95% (38/40)                 │ 97.5%   │    
  ├────────────────────────┼───────────────────┼─────────────────────────────┼─────────┤    
  │ Assembly (peg-in-hole) │ 70% (28/40)       │ 90% (36/40)                 │ 85%     │
  ├────────────────────────┼───────────────────┼─────────────────────────────┼─────────┤    
  │ Average                │ 86.25%            │ 97.5%                       │ 95.62%  │
  └────────────────────────┴───────────────────┴─────────────────────────────┴─────────┘    
                                                                      
  So SplatSim closes ~88% of the gap between simulation and a real-data oracle, without ever
   touching a real demonstration during training.                     
                                                                                            
  Render quality on a held-out test set of 300 robot configurations: PSNR 22.62, SSIM       
  0.7845. Modest by 3DGS-research standards but evidently good enough to fool the policy.
                                                                                            
  Time savings: 3 hours of simulated data collection vs. 20.5 hours of real-world data      
  collection across all tasks (≈ 7× faster).   
                                                                                            
  ---                                                                 
  4. What's in the GitHub repo                 
                                                                                            
  SplatSim/
  ├── configs/                 # object + trajectory configs                                
  ├── scripts/                                                                              
  │   ├── launch_nodes.py                      
  │   ├── run_env_sim.py                                                                    
  │   └── articulated_robot_pipeline.py                               
  ├── splatsim/                # main package                                               
  └── submodules/                                                                           
      ├── gaussian-splatting   # INRIA reference rasterizer                                 
      ├── pybullet                                                                          
      └── gello_software       # GELLO teleop                         
                                                                                            
  Setup:                                                              
  git clone --recursive https://github.com/qureshinomaan/SplatSim                           
  conda create -n splatsim python=3.12 -y                                                   
  conda activate splatsim                      
  # torch 2.7.1+ with CUDA 12.6 (per their README)                                          
  pip install torch torchvision                                       
  # then pip install each submodule                                                         
                                                                                            
  Pretrained assets (3 Google Drive downloads, linked in README):                           
  - COLMAP outputs for the iPhone scan                                                      
  - Pre-trained Gaussian Splat of the example scene                                         
  - GELLO demonstration trajectories for the apple-on-plate task                            
                                                                                            
  So you can in principle download → run sim → see it work on apple-on-plate without        
  retraining anything, then plug in your own scene later.                                   
                                                                                            
  Important caveats:                                                                        
  - Only UR5 is wired up out of the box. Other arms are "supported via URDF" but you'll need
   to redo the per-link segmentation pipeline.                                              
  - README focuses on trajectory replay + behavioral cloning. Diffusion Policy training/eval
   is referenced as the paper method but the repo's polished path is BC; DP integration is  
  "TODO" in places.                                                                         
  - No Docker. Setup is hand-rolled.           
  - License unspecified in the README sections we read.                                     
                                                                                            
  ---                                                                                       
  5. How it would slot into your setup                                                      
                                                                                            
  You're well-positioned because:                                     
                                                                                            
  - You already have 2 RTX A6000s (way overpowered for SplatSim's RTX 3080 Ti baseline).    
  - You have an existing real2sim/ Nerfstudio playground — its .ply exports could substitute
   for the iPhone-scan asset.                                                               
  - You have RealSense workflows in live_camera/ (the file you opened earlier —
  realsense_basic.py).                                                                      
  - Your boxer/, lingbot-map/ look like robot-related repos — possibly a UR5 or similar arm?
   If so, SplatSim is a near-drop-in.                                                       
                                                                      
  Things you'd need beyond what's already on disk:                                          
  - A UR5 (or commit to porting the per-link segmentation to your robot's URDF — that's the
  real engineering cost).                                                                   
  - Their gaussian-splatting-wrapper submodule (or substitute gsplat/Splatfacto rendering —
  would take a small adapter).                                                              
  - The GELLO arm if you want their teleop pipeline, otherwise you can collect demos via    
  PyBullet GUI.                                
                                                                                            
  ---                                                                 
  6. Honest assessment                                                                      
                                                                      
  Strengths                                    
  - Strongest published "policy trained in sim, deployed zero-shot to real arm via 3DGS"
  result with public code.                                                                  
  - Pure RGB — no depth, no markers, no extra sensors.
  - Complete recipe: scan, segment, simulate, train DP, deploy.                             
                                                                                            
  Weaknesses                                                                                
  - Single robot (UR5) and single 3DGS backbone (INRIA reference) — both quite dated by 2026
   standards (gsplat is faster, FastGS is much faster).                                     
  - Scene authoring is manual: per-link cloud segmentation + ICP is not push-button.
  - Render PSNR (22.6) is much lower than a properly-trained Splatfacto on the same data —  
  the bottleneck is the manual segmentation/recomposition, not the rasterizer.              
  - Static backgrounds only. No deformable / soft-body / fluid.                             
                                                                                            
  Where to go next from SplatSim                                                            
  - For soft-body → Real-to-Sim Policy Eval (paper has soft-body support).
  - For more robots → fork SplatSim, port the segmentation pipeline to a Franka/xArm URDF.  
  - For better rendering → swap the INRIA wrapper for gsplat or a Splatfacto export.      
                                                                                            
  Want me to: (a) clone it into tests/repos/splatsim/, set up the env, and try the          
  apple-on-plate demo? Or (b) start by exporting your existing                              
  real2sim/outputs/truck/splatfacto/ checkpoint to .ply so you have a Gaussian asset ready  
  for whatever robot scene you build first?  