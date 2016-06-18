import pickle

if __name__ == '__main__':
	with open('id_generator.bin', 'rb') as f:
		print(pickle.load(f))