import pickle as pk
import sklearn

try:
    dtc, le = pk.load(open('./chatbot_modelsave', 'rb'))
except ValueError as e:
    if 'incompatible dtype' in str(e):
        raise ValueError("Incompatible data types detected in the pickle file. "
                         "This is likely due to a version mismatch between the scikit-learn used to create the model and the current version. "
                         "Please ensure you are using the same version of scikit-learn.") from e
    else:
        raise
