def print_flags(flags):
	for key, value in vars(flags).items():
		print("{}: {}".format(key, str(value)))
