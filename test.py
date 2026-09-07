"""Backward-compatible alias for the maintained SUES-200 evaluator."""

from test_and_evaluate import eval_and_test, parse_opt


if __name__ == "__main__":
    options = parse_opt(True)
    eval_and_test(options.cfg, options.name, options.seq, options.dist, options.checkpoint)
