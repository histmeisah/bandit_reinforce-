"""
Launch script for Bandit-REINFORCE++ training on GSM8K with diverse prompts.

This script:
1. Loads GSM8K configuration via Hydra
2. Initializes BanditAgenticPipeline with prompt selection
3. Runs training with integrated NeuralUCB prompt selection

Usage:
    python experiments/bandit_math_reasoning/start_bandit_math_reasoning.py

    # With config overrides:
    python experiments/bandit_math_reasoning/start_bandit_math_reasoning.py \
        rollout_batch_size=32 \
        max_steps=1000 \
        bandit_config.exploration_param=1.5
"""

import argparse
from dacite import from_dict
from hydra import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.agentic.agentic_config import AgenticConfig
from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline
from roll.utils.logging import get_logger

logger = get_logger()

# Global bandit_actor registry for environments to access
GLOBAL_BANDIT_ACTOR = None


def main():
    parser = argparse.ArgumentParser(description="Bandit-GSM8K Training with Diverse Prompts")
    parser.add_argument(
        "--config_path",
        help="The path of the main configuration file",
        default="../../experiments/bandit_math_reasoning"
    )
    parser.add_argument(
        "--config_name",
        help="The name of the main configuration file (without extension)",
        default="config"
    )
    args = parser.parse_args()

    # Initialize Hydra
    initialize(config_path=args.config_path, job_name="bandit_gsm8k")
    cfg = compose(config_name=args.config_name)

    logger.info("=" * 80)
    logger.info("Bandit-REINFORCE++ GSM8K Training with Diverse Prompts")
    logger.info("=" * 80)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # IMPORTANT: Setup bandit components BEFORE creating AgenticConfig
    # so they can be injected into env_config before make_env_configs() is called
    bandit_components = None
    if hasattr(cfg, 'bandit_config') and cfg.bandit_config is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Pre-initializing Bandit components...")
        logger.info("=" * 80)

        from roll.algorithms.bandit.prompt_loader import PromptLoader
        from roll.algorithms.bandit.bandit_actor import BanditActor

        # Load prompts
        loader = PromptLoader(cfg.bandit_config.prompt_config_path)
        prompt_templates = loader.get_preset_prompts(cfg.bandit_config.preset_name)
        logger.info(f"Loaded {len(prompt_templates)} prompts: {[p.name for p in prompt_templates]}")

        # Load encoder
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(cfg.bandit_config.encoder_model)
            test_emb = encoder.encode("test")
            logger.info(f"Loaded encoder: {cfg.bandit_config.encoder_model} (dim={test_emb.shape[0]})")
        except Exception as e:
            logger.warning(f"Failed to load encoder: {e}")
            encoder = None

        bandit_components = {
            'prompt_templates': prompt_templates,
            'problem_encoder': encoder,
            'bandit_config': cfg.bandit_config
        }

        # Inject into env_config BEFORE creating AgenticConfig
        cfg_container = OmegaConf.to_container(cfg, resolve=True)
        for env_name, env_cfg in cfg_container['custom_envs'].items():
            if 'math_reasoning_bandit' in str(env_cfg.get('env_type', '')).lower():
                if 'env_config' not in env_cfg:
                    env_cfg['env_config'] = {}
                env_cfg['env_config']['prompt_templates'] = prompt_templates
                env_cfg['env_config']['problem_encoder'] = encoder
                env_cfg['env_config']['bandit_actor'] = None  # Will be set after Ray init
                logger.info(f"Injected bandit components into {env_name}")

        logger.info("=" * 80 + "\n")
        cfg = OmegaConf.create(cfg_container)

    # Convert to AgenticConfig
    ppo_config = from_dict(data_class=AgenticConfig, data=OmegaConf.to_container(cfg, resolve=True))

    # Initialize Ray
    init()

    # Create AgenticPipeline (regular pipeline, not Bandit-specific)
    logger.info("\nInitializing AgenticPipeline...")
    pipeline = AgenticPipeline(pipeline_config=ppo_config)
    logger.info("Pipeline initialized successfully!")

    # Now create BanditActor and register globally
    if bandit_components is not None:
        global GLOBAL_BANDIT_ACTOR

        logger.info("Creating BanditActor...")
        bandit_cfg = bandit_components['bandit_config']
        prompt_templates = bandit_components['prompt_templates']

        context_dim = bandit_cfg.get("context_dim", 768)
        bandit_actor = BanditActor.remote(
            n_prompts=len(prompt_templates),
            context_dim=context_dim,
            hidden_dims=bandit_cfg.get("hidden_dims", [512, 256]),
            exploration_param=bandit_cfg.get("exploration_param", 1.0),
            neural_ucb_kwargs={
                "learning_rate": bandit_cfg.get("learning_rate", 0.001),
                "reg_param": bandit_cfg.get("reg_param", 1.0),
                "buffer_size": bandit_cfg.get("buffer_size", 10000),
                "batch_size": bandit_cfg.get("batch_size", 32),
                "update_freq": bandit_cfg.get("update_freq", 10),
                "seed": ppo_config.seed,
            },
            prompt_names=[p.name for p in prompt_templates],
            enable_monitoring=bandit_cfg.get("enable_monitoring", True),
        )
        logger.info("BanditActor created")

        # Register globally
        GLOBAL_BANDIT_ACTOR = bandit_actor

        # Store bandit info in pipeline for monitoring
        pipeline.bandit_actor = bandit_actor
        pipeline.prompt_templates = prompt_templates
        pipeline.problem_encoder = bandit_components['problem_encoder']

    # Print bandit configuration summary
    if hasattr(pipeline, 'bandit_actor') and pipeline.bandit_actor is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Bandit Configuration Summary")
        logger.info("=" * 80)
        logger.info(f"Number of prompts: {len(pipeline.prompt_templates)}")
        logger.info(f"Prompt names: {[p.name for p in pipeline.prompt_templates]}")
        if hasattr(cfg, 'bandit_config'):
            logger.info(f"Encoder: {cfg.bandit_config.encoder_model}")
            logger.info(f"Context dimension: {cfg.bandit_config.context_dim}")
            logger.info(f"Exploration parameter (alpha): {cfg.bandit_config.exploration_param}")
        logger.info("=" * 80 + "\n")
    else:
        logger.warning("\n⚠️  Bandit components not initialized! Check configuration.\n")

    # Run training
    logger.info("Starting training...\n")
    try:
        pipeline.run()
        logger.info("\nTraining completed successfully!")
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}", exc_info=True)
        raise

    # Print final bandit summary
    if hasattr(pipeline, 'bandit_actor') and pipeline.bandit_actor is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Final Bandit Statistics")
        logger.info("=" * 80)
        pipeline.print_bandit_summary()

    logger.info("\n" + "=" * 80)
    logger.info("Training session ended")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
