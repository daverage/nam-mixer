from hybrid.training.epoch_progress import parse_epoch_progress


def test_marker_is_one_based_and_wins_over_lightning_bar():
    # The real end of a 20-epoch local run: our marker, then Lightning's
    # zero-based final Rich bar.
    log = (
        "--epoch-preset=draft: training for 20 epochs.\n"
        "NAM Mixer: epoch 19 of 20\nNAM Mixer: epoch 20 of 20\n"
        "`Trainer.fit` stopped: `max_epochs=20` reached.\n"
        "Epoch 19/19 ━━━━━━━━━━ 62/62 0:00:06 • 0:00:00 10.16it/s v_num: 0.000\n"
    )
    assert parse_epoch_progress(log) == {"epoch": 20, "total_epochs": 20}


def test_marker_reports_the_running_epoch():
    assert parse_epoch_progress("NAM Mixer: epoch 1 of 20\n") == {"epoch": 1, "total_epochs": 20}


def test_lightning_rich_bar_is_converted_from_zero_based():
    assert parse_epoch_progress("Epoch 19/19 ━━━ 62/62") == {"epoch": 20, "total_epochs": 20}
    assert parse_epoch_progress("Epoch 0/19 ━━━ 1/62") == {"epoch": 1, "total_epochs": 20}


def test_lightning_tqdm_bar_is_converted_and_paired_with_preset_total():
    log = "--epoch-preset=draft: training for 20 epochs.\nEpoch 0: 40%|####"
    assert parse_epoch_progress(log) == {"epoch": 1, "total_epochs": 20}


def test_unparseable_is_none():
    assert parse_epoch_progress("") is None
    assert parse_epoch_progress(None) is None
    assert parse_epoch_progress("Preparing data") is None
