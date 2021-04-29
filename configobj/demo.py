import argparse
from pathlib import Path
from configobj import ConfigObj
from validate import Validator

def parse_args():
    parser = argparse.ArgumentParser(description="iris classifier")

    parser.add_argument("config_path", type=str)

    args = parser.parse_args()
    return args

def read_config(config_path):
	config_type = Path(config_path).parts[-2]
	spec_path = Path(config_path).parent.joinpath('configspec.ini')

	cfgspec = ConfigObj(str(spec_path), _inspec=True)
	cfg = ConfigObj(config_path, configspec=cfgspec, )
	res = cfg.validate(Validator())

	_validate_result(res)
	_dict_nulls_to_none(cfg)

	return cfg, config_type

def choose_parser(config, config_type):
	exec("_parse_" + str(config_type).split("_config")[0] + "(config)")

def _validate_result(result):
	if result is not True:
		for key, value in result.items():
			if isinstance(value, dict):
				_validate_result(value)
			if value is not True:
				raise RuntimeError('Config parameter "%s": %s' % (key, value))

def _dict_nulls_to_none(d):
	for k, v in d.items():
		if isinstance(v, dict):
			_dict_nulls_to_none(v)
		elif v in ['null', 'None']:
			d[k] = None

def _parse_sports_team(config):
	print()
	print(config["area"] + " " + config["name"])
	print("Founded: " + str(config["history"]["founded"]))
	print("\nStarters:\n")

	players = [config["players"][key] for key in config["players"]]
	for player in players:
		if player["starter"]:
			print(player["name"] + " " + player["position"])

	print()

if __name__ == "__main__":
    args = parse_args()
    cfg, cfg_type = read_config(args.config_path)
    choose_parser(cfg, cfg_type)