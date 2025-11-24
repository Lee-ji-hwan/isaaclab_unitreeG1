import torch
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg

# --------------------------------------------------------
# Reward Functions
# --------------------------------------------------------

def reward_dynamic_squat(env):
    """
    Deep & Slow Squat Reward
    - 목표: 사인파(Sine Wave)를 그리며 깊게 앉았다 일어나기
    - 범위: 0.44m ~ 0.76m (G1 로봇 관절 한계 고려)
    - 속도: 0.7Hz (안정적인 양발 동기화를 위해 저속 설정)
    """
    current_time = env.episode_length_buf * env.step_dt
    
    freq = 0.1
    target_height = 0.60 + 0.16 * torch.sin(freq * current_time)
    
    root_height = env.scene["robot"].data.root_pos_w[:, 2]
    error = torch.abs(root_height - target_height)
    
    # 오차가 적을수록 지수적으로 높은 보상 (판정 범위 0.05)
    return torch.exp(-torch.square(error) / (0.05**2))


def reward_double_support(env):
    """
    Double Support Reward
    - 목표: 양발이 동시에 지면에 닿아있도록 유도 (짝다리 방지)
    """
    # 발바닥 접촉 센서 확인 (> 1.0이면 접촉)
    contact = env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, 2] > 1.0
    num_contacts = torch.sum(contact, dim=1)
    
    # 두 발이 모두 닿았을 때만 1.0 반환
    return (num_contacts == 2).float()


# --------------------------------------------------------
# Environment Configuration
# --------------------------------------------------------

@configclass
class G1SquatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ======================================================
        # 1. Scene & Terrain Settings (평지 설정)
        # ======================================================
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # ======================================================
        # 2. Commands (제자리 고정)
        # ======================================================
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # ======================================================
        # 3. Rewards (3대 핵심 조건)
        # ======================================================
        
        # [Core 1] Deep Squat Performance
        self.rewards.dynamic_squat = RewTerm(
            func=reward_dynamic_squat,
            weight=5.0  # 가장 높은 가중치 (메인 목표)
        )

        # [Core 2] Stability & Symmetry (양발 착지)
        self.rewards.double_support = RewTerm(
            func=reward_double_support,
            weight=2.0  # 짝다리 방지 및 안정성 확보
        )

        # [Core 3] Zero Velocity Constraints (위치 고정)
        self.rewards.track_lin_vel_xy_exp.weight = 1.5  # XY 이동 억제
        self.rewards.track_ang_vel_z_exp.weight = 1.0   # 회전 억제

        # [Constraints] 발바닥 고정 (Sliding & Jumping 방지)
        self.rewards.feet_slide.weight = -1.0           # 미끄러짐 벌점 (적정 수준)
        self.rewards.feet_air_time.weight = -1.0        # 발 떼기 벌점
        self.rewards.feet_air_time.params["threshold"] = 0.1

        # [Damping] 동작 안정화 (떨림 방지)
        self.rewards.action_rate_l2.weight = -0.005     # 급격한 관절 변화 억제
        self.rewards.dof_acc_l2.weight = -2.5e-7        # 급가속 억제