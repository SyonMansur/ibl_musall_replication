import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV # ridge cross val to try different alpha values 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d # continuous firing rate so the model can predict (gaussian smoothing)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.loader import MusallDataLoader


# goal here is to do the same toeplitz matrix as in the original musall paper
# shift the velocity vector and stack them
# now, row t has the entire history and future window of movement
# separate weight for every time lag (gets feedback and prediction)

# basically, create fake features that represent the past and the future
def create_temporal_design_matrix(vel, lags=20): # Increased to +/- 1.0 second
    n_samples = len(vel)
    padded = np.pad(np.abs(vel), lags, mode='constant') # add zeros at the start and the end so we can slide window 
    design_matrix = []
    for i in range(2 * lags + 1):
        design_matrix.append(padded[i : i + n_samples]) # take the entire velocity array, shift it one step, then save as a new feature
    return np.stack(design_matrix, axis=1) # glue the shifted arrays together side by side

# now the ridge regression can assign a weight to every column, turning time into space
# basically a CNN but manually

def run_analysis():
    # load Data
    eid = '4b00df29-3769-43be-bb40-128b1cba6d35'
    loader = MusallDataLoader()
    vel, neural_matrix = loader.process_sessions(eid) # (time, neurons)

    # make sure that interp1d and histogram2d agree-- cut the longer one to match if needed
    min_len = min(len(vel), neural_matrix.shape[0])
    vel = vel[:min_len]
    neural_matrix = neural_matrix[:min_len]

    # smoothing - binary spikes into a continunuous probability rate
    neural_matrix_smooth = gaussian_filter1d(neural_matrix, sigma=2, axis=0)

    # make the toeplitz
    X = create_temporal_design_matrix(vel, lags=20)
    
    # split and test
    X_train, X_test, y_train_all, y_test_all = train_test_split(
        X, neural_matrix_smooth, test_size=0.2, shuffle=False
    ) # NO SHUFFLING (time based events)

    # find best alpha
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0])
    model.fit(X_train, y_train_all)
    
    print(f"optimal alpha: {model.alpha_}")

    # eval
    y_pred_all = model.predict(X_test)
    
    # get r2
    # no model.score because we don't want to get the average r2, we want the individual neurons
    ss_res = np.sum((y_test_all - y_pred_all) ** 2, axis=0) # residual sum of squares
    ss_tot = np.sum((y_test_all - np.mean(y_test_all, axis=0)) ** 2, axis=0) # total sum of squares
    scores = 1 - (ss_res / (ss_tot + 1e-8)) # r2 for every unit
    
    best_index = np.argmax(scores) # get max
    best_score = scores[best_index]
    
    print("results:")
    print(f"best neuron: {best_index}")
    print(f"best r2:  {best_score:.4f}")

    # plot
    plt.figure(figsize=(15, 6))
    vel_test = vel[len(vel)-len(y_test_all):]
    peak_move_idx = np.argmax(np.abs(vel_test))
    plot_slice = slice(max(0, peak_move_idx - 100), min(len(y_test_all), peak_move_idx + 100))
    
    plt.subplot(2, 1, 1)
    plt.title(f"best neuron {best_index} (r2={best_score:.4f})")
    plt.plot(vel_test[plot_slice], color='green', label='Velocity')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(y_test_all[plot_slice, best_index], color='black', alpha=0.6, label='actual')
    plt.plot(y_pred_all[plot_slice, best_index], color='red', linewidth=2, label='ridge pred')
    plt.legend()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/optimized_ridge.png')
    plt.show()

if __name__ == "__main__":
    run_analysis()