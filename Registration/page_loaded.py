import pickle, sys

if __name__ == '__main__':
	if sys.argv[1] == "init":
		with open('page4_loaded.bin', 'wb') as f:
			pickle.dump("False", f)
	elif sys.argv[1] == "check":
		with open('page4_loaded.bin', 'rb') as f:
			print(pickle.load(f))
		with open('page4_loaded.bin', 'wb') as f:
			pickle.dump("True", f)