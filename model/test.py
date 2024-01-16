import pickle


with open('data.pkl', 'rb') as f:
    img_arr = pickle.load(f)

print(img_arr)