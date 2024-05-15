import matplotlib.pyplot as plt

def plot_embeddings(df):
    fig, ax = plt.subplots()
    ax.scatter(df['x'], df['y'])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D projection of docs')
    return fig
