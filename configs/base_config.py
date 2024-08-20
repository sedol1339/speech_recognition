from ml_collections.config_dict import ConfigDict

# TODO migrate to MMEngine configs


def get_config():
    config = ConfigDict()

    config.exit_after_downloading = False

    config.device = 'cuda'
    config.seed = 0

    config.model = ConfigDict()
    config.model.type = 'wav2vec2'
    config.model.path = 'emre/wav2vec2-xls-r-300m-Russian-small'
    config.model.load_optimizer_if_saved = True

    config.whisper = ConfigDict()
    config.whisper.kwargs = ConfigDict()

    config.wav2vec2 = ConfigDict()
    config.wav2vec2.kwargs = ConfigDict()
    config.wav2vec2.kwargs.add_ru_vocab = False
    config.wav2vec2.kwargs.input_text_case = 'lower'
    config.wav2vec2.kwargs.drop_unk_from_input_text = True
    config.wav2vec2.kwargs.model_kwargs = ConfigDict()
    config.wav2vec2.kwargs.model_kwargs.apply_spec_augment = None
    config.wav2vec2.kwargs.model_kwargs.activation_dropout = None
    config.wav2vec2.kwargs.model_kwargs.attention_dropout = None
    config.wav2vec2.kwargs.model_kwargs.feat_proj_dropout = None
    config.wav2vec2.kwargs.model_kwargs.final_dropout = None
    config.wav2vec2.kwargs.model_kwargs.hidden_dropout = None
    config.wav2vec2.kwargs.model_kwargs.hidden_dropout_prob = None
    config.wav2vec2.kwargs.model_kwargs.layerdrop = None
    config.wav2vec2.kwargs.model_kwargs.mask_time_prob = None
    config.wav2vec2.kwargs.pad_to_seconds = None

    config.data = ConfigDict()
    config.data.min_seconds = 1
    config.data.max_seconds = 10

    config.data.train = ConfigDict()
    config.data.train.max_samples = None
    config.data.train.batch_size = 16
    config.data.train.datasets = (
        'rulibrispeech:train',  # 54K samples
        # 'sova_rudevices:train',  # 74K samples
        # 'common_voice_17_0:train',  # 26K samples
    )

    config.data.eval = ConfigDict()
    config.data.eval.max_samples = 200
    config.data.eval.batch_size = 32
    config.data.eval.datasets = (
        'rulibrispeech:test',
        'sova_rudevices:test',
        'common_voice_17_0:test',
        'sova_ruyoutube:val',
        'golos_farfield:test',
        'resd:test',
        'fleurs:test',
        'speech_massive_test:test',
        'taiga_speech_v2:train',
    )
    config.data.eval.add_train_datasets = True

    config.optimizer = ConfigDict()
    config.optimizer.weight_decay = 1e-4  # gets multiplied by lr
    config.optimizer.lr = 1e-4

    config.birm = ConfigDict()
    config.birm.enable = False
    config.birm.start_from_step = 0
    config.birm.type = 'auto'
    config.birm.samples_number = None
    config.birm.scaling_type = 'relative'
    config.birm.scale = 0.1

    config.training = ConfigDict()
    config.training.loss_scale = 0.01
    config.training.max_steps = 20_000

    config.training.gradient_clipping = ConfigDict()
    config.training.gradient_clipping.enabled = True
    config.training.gradient_clipping.max_norm = 10

    config.analysis = ConfigDict()
    config.analysis.nan_logits = True
    config.analysis.nan_loss = False
    config.analysis.non_finite_grad = True
    config.analysis.on_fail_save_current_step = True
    config.analysis.weights_tracker = ConfigDict()
    config.analysis.weights_tracker.enabled = True
    config.analysis.weights_tracker.n_elements = 50
    config.analysis.weights_tracker.n_elements_override = {
        'model:lm_head:bias': 200,
    }

    config.evaluation = ConfigDict()
    config.evaluation.enabled = True
    config.evaluation.calc_loss = True
    config.evaluation.steps = ConfigDict()
    config.evaluation.steps.every_n_steps = 500
    config.evaluation.steps.log_scale = True
    config.evaluation.steps.log_scale_base = 1.3
    config.evaluation.steps.log_scale_min_delta = 3

    config.saving = ConfigDict()
    config.saving.name = ''
    config.saving.evals_dir = 'results'
    config.saving.save_model = False
    config.saving.save_optimizer = False

    return config
