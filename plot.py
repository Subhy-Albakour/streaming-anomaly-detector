import pandas as pd
import matplotlib.pyplot as plt





def plot_results(filename,models_names,title="Outlier Detection"):
    df = pd.read_csv(filename, comment='#')
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(title, fontsize=14)
    mean_acc=["mean_acc_[M%s]"%(str(i)) for i in range(len(models_names))]
    curr_acc=["current_acc_[M%s]"%(str(i)) for i in range(len(models_names))]
    mean_kappa=["mean_kappa_[M%s]"%(str(i)) for i in range(len(models_names))]
    curr_kappa=["current_kappa_[M%s]"%(str(i)) for i in range(len(models_names))]
    ax = df.plot(ax=axes[0,0],x="id", y=mean_acc, rot=45, linewidth=3, title="Mean Accuracy")
    ax.legend(models_names, loc='best')
    ax = df.plot(ax=axes[0,1],x="id", y=curr_acc, rot=30, linewidth=3, title="Current Accuracy")
    ax.legend(models_names, loc='best')
    ax = df.plot(ax=axes[1,0],x="id", y=mean_kappa, rot=45, linewidth=3)
    ax.legend(models_names, loc='best')
    ax.set_xlabel("Mean Kappa")
    ax = df.plot(ax=axes[1,1],x="id", y=curr_kappa, rot=30, linewidth=3,)
    ax.set_xlabel("Current Kappa")
    ax.legend(models_names, loc='best')
    plt.savefig(filename+".png")
    
    plt.show()

if __name__ == "__main__":
#,"MiniBatchKMeans","MiniBatchKMeans(X)"
    plot_results("result.csv",["BICO","BICO(X)","Birch","MB","MBX"])