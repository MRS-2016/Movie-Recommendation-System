import pickle

if __name__ == '__main__':
	with open('id_generator.bin', 'rb') as f:
		_id = pickle.load(f)
		print(_id)
	with open('id_generator.bin', 'wb') as f:
		_id += 1
		pickle.dump(_id, f)