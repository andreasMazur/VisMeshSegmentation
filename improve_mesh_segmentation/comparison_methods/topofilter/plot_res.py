import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_compare():
    params = "15_5_32_5"
    data_imcnn = np.load(f"/tmp/imcnn_stats_{params}.npy", allow_pickle=True).item()
    data       = np.load(f"./stats_{params}.npy", allow_pickle=True).item()

    train_set_size    = np.array(data['train_set_size'])
    clean_in_set      = np.array(data['clean_in_set'])
    noisy_in_set      = np.array(data['noisy_in_set'])
    noisy_outside_set = np.array(data['noisy_outside_train'])
    clean_outside_set = np.array(data['clean_outside_train'])
    val_acc           = np.array(data['val_acc'])
    train_acc         = np.array(data['train_acc'])

    out_set_size = clean_outside_set + noisy_outside_set
    total_size   = train_set_size + out_set_size

    scale_size = total_size

    x = np.arange(len(train_set_size))
    
    fig, axs = plt.subplots(3, 2, sharey='row', sharex='col')
    axs[0, 0].plot(x, clean_in_set/scale_size, label='Clean in train set')
    axs[0, 0].plot(x, noisy_in_set/scale_size, label='Noisy in train set')
    axs[0, 0].plot(x, noisy_outside_set/scale_size, label='Noisy outside train set')
    axs[0, 0].plot(x, clean_outside_set/scale_size, label='Clean outside train set')

    axs[0, 0].set_ylabel('Percentage w.r.t. Full Set')
    axs[0, 0].legend()

    scale_size = train_set_size
    axs[1, 0].plot(x, clean_in_set/scale_size, label='Clean in train set')
    axs[1, 0].plot(x, noisy_in_set/scale_size, label='Noisy in train set')
    axs[1, 0].plot(x, noisy_outside_set/out_set_size, label='Noisy outside train set')
    axs[1, 0].plot(x, clean_outside_set/out_set_size, label='Clean outside train set')

    axs[1, 0].set_ylabel('Percentage in Respective Set')
    axs[1, 0].legend()

    axs[2, 0].plot(x, val_acc, label='Validation Accuracy')
    axs[2, 0].plot(x, train_acc, label='Train Accuracy')
    axs[2, 0].set_xlabel('Iteration')
    axs[2, 0].set_ylabel('Accuracy')
    axs[2, 0].legend()

    train_set_size    = np.array(data_imcnn['train_set_size'])
    clean_in_set      = np.array(data_imcnn['clean_in_set'])
    noisy_in_set      = np.array(data_imcnn['noisy_in_set'])
    noisy_outside_set = np.array(data_imcnn['noisy_outside_train'])
    clean_outside_set = np.array(data_imcnn['clean_outside_train'])
    val_acc           = np.array(data_imcnn['val_acc'])
    train_acc         = np.array(data_imcnn['train_acc'])

    out_set_size = clean_outside_set + noisy_outside_set
    total_size   = train_set_size + out_set_size

    scale_size = total_size

    x = np.arange(len(train_set_size))
    
    axs[0, 1].plot(x, clean_in_set/scale_size, label='Clean in train set')
    axs[0, 1].plot(x, noisy_in_set/scale_size, label='Noisy in train set')
    axs[0, 1].plot(x, noisy_outside_set/scale_size, label='Noisy outside train set')
    axs[0, 1].plot(x, clean_outside_set/scale_size, label='Clean outside train set')

    axs[0, 1].set_ylabel('Percentage w.r.t. Full Set')
    axs[0, 1].legend()

    scale_size = train_set_size
    axs[1, 1].plot(x, clean_in_set/scale_size, label='Clean in train set')
    axs[1, 1].plot(x, noisy_in_set/scale_size, label='Noisy in train set')
    axs[1, 1].plot(x, noisy_outside_set/out_set_size, label='Noisy outside train set')
    axs[1, 1].plot(x, clean_outside_set/out_set_size, label='Clean outside train set')

    axs[1, 1].set_ylabel('Percentage in Respective Set')
    axs[1, 1].legend()

    axs[2, 1].plot(x, val_acc, label='Validation Accuracy')
    axs[2, 1].plot(x, train_acc, label='Train Accuracy')
    axs[2, 1].set_xlabel('Iteration')
    axs[2, 1].set_ylabel('Accuracy')
    axs[2, 1].legend()

    axs[0, 0].set_title("PointNet")
    axs[0, 1].set_title("IMCNN")
    fig.suptitle("KNN params: " + params)

    plt.show()

def plot_single():
    data = np.load("/tmp/imcnn_stats_15_5_32_100.npy", allow_pickle=True).item()

    train_set_size = np.array(data['train_set_size'])
    clean_in_set = np.array(data['clean_in_set'])
    noisy_in_set = np.array(data['noisy_in_set'])
    noisy_outside_set = np.array(data['noisy_outside_train'])
    clean_outside_set = np.array(data['clean_outside_train'])
    val_acc = np.array(data['val_acc'])
    train_acc = np.array(data['train_acc'])

    out_set_size = clean_outside_set + noisy_outside_set
    total_size = train_set_size + out_set_size

    scale_size = total_size

    x = np.arange(len(train_set_size))
    
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(x, clean_in_set/scale_size, label='Clean in train set')
    axs[0].plot(x, noisy_in_set/scale_size, label='Noisy in train set')
    axs[0].plot(x, noisy_outside_set/scale_size, label='Noisy outside train set')
    axs[0].plot(x, clean_outside_set/scale_size, label='Clean outside train set')

    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Percentage w.r.t. Full Set')
    axs[0].legend()

    scale_size = train_set_size
    axs[1].plot(x, clean_in_set/scale_size, label='Clean in train set')
    axs[1].plot(x, noisy_in_set/scale_size, label='Noisy in train set')
    axs[1].plot(x, noisy_outside_set/out_set_size, label='Noisy outside train set')
    axs[1].plot(x, clean_outside_set/out_set_size, label='Clean outside train set')

    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('Percentage in Respective Set')
    axs[1].legend()

    # axs[1].plot(x, train_set_size, label='Train Set Size')
    # axs[1].plot(x, noisy_outside_set+clean_outside_set, label='Filtered out Set Size')
    # axs[1].legend()
    # axs[1].set_xlabel('Iteration')
    # axs[1].set_ylabel('Number of Vertices')
    # print("Train Set Size:", train_set_size)
    # print("Filtered out Set Size:", noisy_outside_set + clean_outside_set)
    # axs[1].set_yscale('log')

    axs[2].plot(x, val_acc, label='Validation Accuracy')
    axs[2].plot(x, train_acc, label='Train Accuracy')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('Accuracy')
    axs[2].legend()

    plt.show()

def compare_params():
    fig, axs = plt.subplots(2, 1, sharex='col')

    for k_cc in [10, 25, 50, 100]:
        params = f"15_5_32_{k_cc}"
        data_imcnn = np.load(f"/tmp/imcnn_stats_{params}.npy", allow_pickle=True).item()

        train_set_size    = np.array(data_imcnn['train_set_size'])
        clean_in_set      = np.array(data_imcnn['clean_in_set'])
        noisy_in_set      = np.array(data_imcnn['noisy_in_set'])
        noisy_outside_set = np.array(data_imcnn['noisy_outside_train'])
        clean_outside_set = np.array(data_imcnn['clean_outside_train'])
        val_acc           = np.array(data_imcnn['val_acc'])
        train_acc         = np.array(data_imcnn['train_acc'])

        out_set_size = clean_outside_set + noisy_outside_set
        total_size   = train_set_size + out_set_size

        scale_size = total_size

        x = np.arange(len(train_set_size))
        
        axs[0].plot(x, clean_in_set/scale_size, label=params) 
        axs[1].plot(x, noisy_in_set/scale_size, label=params)
    axs[0].set_title("Clean in train set")
    axs[1].set_title("Noisy in train set")
        # axs[0].plot(x, noisy_outside_set/scale_size, label='Noisy outside train set')
        # axs[0].plot(x, clean_outside_set/scale_size, label='Clean outside train set')
    axs[0].legend()
    axs[1].legend()


def compare_params_models():
    fig, axs = plt.subplots(2, 2, sharex='col')

    for k_cc in [10, 25, 50, 100]:#, 250]:
        params = f"15_10_32_{k_cc}"
        data = np.load(f"./stats_{params}.npy", allow_pickle=True).item()

        train_set_size    = np.array(data['train_set_size'])
        clean_in_set      = np.array(data['clean_in_set'])
        noisy_in_set      = np.array(data['noisy_in_set'])
        noisy_outside_set = np.array(data['noisy_outside_train'])
        clean_outside_set = np.array(data['clean_outside_train'])
        val_acc           = np.array(data['val_acc'])
        train_acc         = np.array(data['train_acc'])

        out_set_size = clean_outside_set + noisy_outside_set
        total_size   = train_set_size + out_set_size

        scale_size = total_size

        x = np.arange(len(train_set_size))
        
        axs[0, 0].plot(x, clean_in_set/scale_size, label=params) 
        axs[1, 0].plot(x, noisy_in_set/scale_size, label=params)

        data_imcnn = np.load(f"./imcnn_stats_{params}.npy", allow_pickle=True).item()

        train_set_size    = np.array(data_imcnn['train_set_size'])
        clean_in_set      = np.array(data_imcnn['clean_in_set'])
        noisy_in_set      = np.array(data_imcnn['noisy_in_set'])
        noisy_outside_set = np.array(data_imcnn['noisy_outside_train'])
        clean_outside_set = np.array(data_imcnn['clean_outside_train'])
        val_acc           = np.array(data_imcnn['val_acc'])
        train_acc         = np.array(data_imcnn['train_acc'])

        out_set_size = clean_outside_set + noisy_outside_set
        total_size   = train_set_size + out_set_size

        scale_size = total_size

        x = np.arange(len(train_set_size))
        
        axs[0, 1].plot(x, clean_in_set/scale_size, label=params) 
        axs[1, 1].plot(x, noisy_in_set/scale_size, label=params)

    axs[0, 0].set_ylabel("Clean in train set")
    axs[1, 0].set_ylabel("Noisy in train set")

    axs[0, 0].set_title("PointNet")
    axs[0, 1].set_title("IMCNN")
        # axs[0].plot(x, noisy_outside_set/scale_size, label='Noisy outside train set')
        # axs[0].plot(x, clean_outside_set/scale_size, label='Clean outside train set')
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    plt.show()

if __name__ == "__main__":
    sns.set_theme()
    # plot_single()
    # plot_compare()
    # compare_params()
    compare_params_models()