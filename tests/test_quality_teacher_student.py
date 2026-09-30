from dataclasses import fields

from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.agents.rsl_rl_ppo_cfg import (
    PPORunnerCfg as OriginalTeacherRunnerCfg,
)
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.agents.rsl_rl_quality_ppo_cfg import (
    QualityPPORunnerCfg,
)
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.env_cfg import PeriodicWalkingEnvCfg
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking.quality_env_cfg import QualityTeacherEnvCfg
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.agents import (
    rsl_rl_quality_distillation_cfg as student_quality_agent,
)
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.env_cfg import StudentEnvCfg
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student.quality_env_cfg import (
    QualityStudentEnvCfg,
)
from zbot_rl_isaaclab.tasks.velocity.config.zbot_6dof_periodic_walking_student_finetune.env_cfg import (
    StudentFinetuneEnvCfg,
)


def _terms(group) -> list[str]:
    return [field.name for field in fields(group) if hasattr(getattr(group, field.name), "func")]


def test_quality_teacher_and_student_use_one_reward_objective():
    teacher = QualityTeacherEnvCfg()
    student = QualityStudentEnvCfg()
    nominal = StudentFinetuneEnvCfg()

    assert type(teacher.rewards) is type(student.rewards) is type(nominal.rewards)
    assert teacher.rewards == student.rewards == nominal.rewards
    assert teacher.commands.weight_shift.curriculum_initial_maximum_frequency is None
    assert student.commands.weight_shift.curriculum_initial_maximum_frequency == 1.0
    assert teacher.commands.weight_shift.frequency_reward_term_name == "alternating_touchdown"
    assert student.commands.weight_shift.frequency_reward_term_name == "alternating_touchdown"
    assert teacher.rewards.alternating_touchdown.params["minimum_alternation_interval"] == 0.15


def test_quality_stages_keep_the_existing_teacher_and_student_observation_contracts():
    assert _terms(QualityTeacherEnvCfg().observations.policy) == _terms(PeriodicWalkingEnvCfg().observations.policy)
    assert _terms(QualityStudentEnvCfg().observations.policy) == _terms(StudentEnvCfg().observations.policy)
    assert _terms(QualityStudentEnvCfg().observations.teacher) == _terms(QualityTeacherEnvCfg().observations.policy)

    quality_teacher = QualityPPORunnerCfg()
    original_teacher = OriginalTeacherRunnerCfg()
    quality_student = student_quality_agent.QualityDistillationRunnerCfg()
    assert quality_teacher.actor.hidden_dims == original_teacher.actor.hidden_dims
    assert quality_student.teacher.hidden_dims == quality_teacher.actor.hidden_dims
    assert quality_student.obs_groups == {"student": ["policy"], "teacher": ["teacher"]}
    assert quality_student.student.distribution_cfg.init_std == 0.03
    assert quality_student.save_interval == 25
    assert quality_student.algorithm.learning_rate == 1.0e-4
    assert quality_student.algorithm.teacher_fraction_end == 0.7
    assert quality_student.algorithm.teacher_fraction_decay_updates == 600
