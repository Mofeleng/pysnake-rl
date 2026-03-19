import threading
import matplotlib
import matplotlib.pyplot as plt

_plot_data = {'scores': [], 'mean_scores': []}
_lock = threading.Lock()

def _plot_worker():
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    while True:
        with _lock:
            scores = list(_plot_data['scores'])
            mean_scores = list(_plot_data['mean_scores'])

        ax.cla()
        ax.set_title("Training PySnake RL")
        ax.set_xlabel("Number of Games")
        ax.set_ylabel("Score")
        ax.plot(scores, label="Score")
        ax.plot(mean_scores, label="Mean Score")
        ax.set_ylim(ymin=0)
        ax.legend()

        if scores:
            ax.text(len(scores) - 1, scores[-1], str(scores[-1]))
        if mean_scores:
            ax.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.5)

_thread = threading.Thread(target=_plot_worker, daemon=True)
_thread.start()

def plot(scores, mean_scores):
    with _lock:
        _plot_data['scores'] = list(scores)
        _plot_data['mean_scores'] = list(mean_scores)