import pickle, sys

if __name__ == '__main__':
	with open('id_generator.bin', 'wb') as f:
		_id = int(sys.argv[1])
		pickle.dump(_id, f)