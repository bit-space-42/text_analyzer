import numpy as np
import re
import spacy
import pickle
import time
import collections
import glob
import sys

import matplotlib.pyplot as plt


class Text_Analyzer:

    # Initialize the Text_Complexity object
    # Limit defines the maximum number of characters that will be extracted
    # and analyzed from each text file.
    def __init__(self, limit=10000, enable_gpu=False):
        self.limit = limit
        self.library = None
        self.texts = collections.OrderedDict()

        if enable_gpu:
            spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = self.limit
        self.texts_pos = collections.OrderedDict()

        self.texts_stats = collections.OrderedDict()
        self.texts_dists = collections.OrderedDict()

    # Import .txt text files from the directory 'library' and limit the total
    # All imported texts are stored as strings in the array `texts`
    def import_texts(self, library):
        filenames = glob.glob(library + '*.txt')
        filenames.sort()
        for filename in filenames:
            text = ''
            with open(filename, 'r') as f:
                for line in f:
                    text += f.read()
            name = filename.replace(library, '')
            name = name.replace('.txt', '')

            if self.limit is not None and len(text) > self.limit:
                text = text[:self.limit]

            # Add the text to self.texts
            self.texts[name] = text

            # Instansiate instance of text statistics
            self.texts_stats[name] = collections.OrderedDict({'name': name})

            # Instantiate dictionary of text distributions
            self.texts_dists[name] = collections.OrderedDict({'name': name})

    def load(self, library):
        print('Loading text files from', library, '...')

        # Start tracking time
        tstart = time.time()

        # Import texts from library
        self.import_texts(library)

        # Preprocess texts
        print('... preprocessing texts')
        for name, text in self.texts.items():
            # replace tabs and new lines with spaces
            text = re.sub('[\t\n]+', ' ', text, flags=re.MULTILINE)
            # remove alphanumerics
            text = re.sub(r'[^a-zA-Z0-9_ ,\-—\-\?\.!\"\'\:\“\”\‘\’]',
                          ' ', text, flags=re.MULTILINE)
            # replace extra spaces with single spaces.
            text = re.sub(' +', ' ', text, flags=re.MULTILINE)

            self.texts[name] = text

        # Tag parts of speech
        print('... tagging parts of speech:')
        for name, text in self.texts.items():
            print('    -> tagging ' + name + '...')
            self.texts_pos[name] = self.nlp(text)

        # Print total time taken
        tend = time.time()
        print('... time taken:', round(tend - tstart, 2), 'seconds')

    def analyze(self):
        print('Analyzing text files...')

        # Start tracking time
        tstart = time.time()

        # Save the total number of tokens in text_stats
        for name, text in self.texts_pos.items():
            num_words = 0
            for token in text:
                if token.text not in ['.', '?', '!', ',', '\'', '\"', '-', '—',
                                      '-', ':', '“', '”', '‘', '’']:
                    num_words += 1

            self.texts_stats[name]['num_words'] = num_words

        # Calculate average word lengths with their respective standard
        # deviations
        for name, text in self.texts_pos.items():
            # Calculate all word lengths
            word_lengths = []
            for token in text:
                word_lengths.append(len(token.text))

            # Calcualte distribution for word lengths and store in texts_dists
            word_len_dist = np.histogram(
                word_lengths, bins=range(max(word_lengths)), density=True)
            self.texts_dists[name]['word_len'] = word_len_dist

            # Calculate Standard deviation and median of word lengths and store
            # in texts_stats
            self.texts_stats[name]['word_len_sd'] = np.std(word_lengths)
            self.texts_stats[name]['word_len_mean'] = np.mean(word_lengths)

        # Calculate vocabulary size
        for name, text in self.texts_pos.items():
            # Calculate unique nouns, adjectives, verbs and adverbs
            vocab = []
            vocab_noun = []
            vocab_descriptors = []

            for token in text:
                if token.text not in ['.', '?', '!', ',', '\'', '\"', '-', '—',
                                      '-', ':', '“', '”', '‘', '’']:
                    vocab.append(token.text)
                if token.pos_ in ['PROPN', 'NOUN', 'X']:
                    vocab_noun.append(token.text)
                elif token.pos_ in ['ADJ', 'VERB', 'ADP', ]:
                    vocab_descriptors.append(token.text)

            vocab = set(vocab)
            vocab_noun = set(vocab_noun)
            vocab_descriptors = set(vocab_descriptors)

            # Save vocab sizes in texts_stats
            self.texts_stats[name]['vocab'] = len(vocab)
            self.texts_stats[name]['vocab_noun'] = len(vocab_noun)
            self.texts_stats[name]['vocab_descriptors'] = len(
                vocab_descriptors)

            # Save normalised vocab sizes in texts_stats
            num_words = self.texts_stats[name]['num_words']
            self.texts_stats[name]['vocab_norm'] = len(vocab) / num_words
            self.texts_stats[name]['vocab_noun_norm'] = len(
                vocab_noun) / num_words
            self.texts_stats[name]['vocab_descriptors_norm'] = len(
                vocab_descriptors) / num_words

        # Calculate mean sentence length
        for name, text in self.texts_pos.items():
            # Store every sentence length in a list
            sent_lens = []
            length = 0
            for token in text:
                if token.text in ['.', '?', '!'] and length > 1:
                    sent_lens.append(length)
                    length = 0
                elif token.text not in ['.', '?', '!', ',', '\'', '\"', '-',
                                        '—', '-', ':', '“', '”', '‘', '’']:
                    length += 1

            # Calculate distribution of sentence lengths and store in
            # texts_dists
            sent_len_dist = np.histogram(
                sent_lens, bins=range(max(sent_lens)), density=True)
            self.texts_dists[name]['sent_len'] = sent_len_dist

            # Save mean sentence length in texts_stats
            self.texts_stats[name]['sent_len_mean'] = np.mean(sent_lens)

        # Calculate average dependency depths
        for name, text in self.texts_pos.items():

            # Store the dependency depth of each word in dep_depths
            dep_depths = []
            for token in text:
                if token.text not in ['.', '?', '!', ',', '\'', '\"', '-', '—',
                                      '-', ':', '“', '”', '‘', '’']:
                    dep_depths.append(len([1 for _ in token.ancestors]))

            # Store (normalised) distribution of dependency depths in texts_dists
            dep_depth_dist = np.histogram(
                dep_depths, bins=range(max(dep_depths)), density=True)
            self.texts_dists[name]['dep_depth'] = dep_depth_dist

            # Store mean and standard deviation of dependency depths
            self.texts_stats[name]['dep_depth_mean'] = np.mean(dep_depths)
            self.texts_stats[name]['dep_depth_sd'] = np.std(dep_depths)

        # Print total time taken
        tend = time.time()
        print('... time taken:', round(tend - tstart, 2), 'seconds')

    # Exports statistics of the texts into a CSV file
    def export_statistics_CSV(self, output_filename):
        with open(output_filename, 'w') as f:
            # Extract attributes from a single entry in texts_stats and write
            # them as a CSV header
            names = list(self.texts_stats.keys())
            csv_header = list(self.texts_stats[names[0]].keys())
            csv_header = ','.join(csv_header)
            f.write(csv_header + '\n')

            # Write attributes of every entry in texts_stats to CSV
            for name, stats in self.texts_stats.items():
                # Produce line to be entered in CSV file
                csv_line = ''
                for stat in stats.values():
                    csv_line += str(stat) + ','
                csv_line = csv_line[:-1]

                # Write line in CSV file
                f.write(csv_line + '\n')

    # A particular text can be exported to the folder `output` in order to test
    # the effects of preprocessing.
    def export_text(self, name, text):
        with open('./output/' + name + '.txt', 'w') as f:
            f.write(text)

    # This method can save an object to the folder `pickle` in the Pickle format
    def export_pickle(self, name, object):
        with open("./pickle/" + name + ".pickle", "wb") as f:
            pickle.dump(object, f)

    # This method can load an object from a Pickle file stored in the `pickle`
    # folder
    def import_pickle(self, name):
        object = []
        with open("./pickle/" + name + ".pickle", "rb") as f:
            object = pickle.load(f)

        return object

    # This plots a bar graph with `xticks` labelling the ticks on the x-axis
    def plot_bar(self, xs, ys, xticks=None):
        plt.bar(xs, ys)
        if xticks is None:
            plt.xticks(xs)
        else:
            plt.xticks(xticks)
        plt.plot()

    # This plots a double bar graph
    def plot_double_bar(self, x1s, y1s, x2s, y2s, xticks=None, w=0.3,
                        legend=['First', 'Second'],
                        xlabel='x', ylabel='y',
                        title='Comparative Bar Graph',
                        file_name='untitled'):

        ax1 = plt.axes()
        g1 = ax1.bar(x1s, y1s, width=-w, color='b', align='edge')
        plt.xlabel(xlabel)

        ax2 = ax1.twinx()
        g2 = ax2.bar(x2s, y2s, width=w, color='r', align='edge')

        if xticks is None:
            if len(x1s) > len(x2s):
                plt.xticks(x1s)
            else:
                plt.xticks(x2s)
        else:
            plt.xticks(xticks)

        plt.title(title)
        plt.ylabel(ylabel)
        plt.legend([g1, g2], legend)

        plt.plot()
        plt.savefig('./plots/bar-chart-' + file_name + '.pdf')

    # This plots a multiple bar graph for the statistics of the first 5 texts
    def plot_multi_bar(self, names):
        length = min(len(filenames), 5)
        x_labels = ['Word Length', 'Vocabulary', 'Grammar', 'Overall']

        # Set plot parameters
        fig, ax = plt.subplots()
        width = 1. / (length + 2)  # width of bar
        x = np.arange(4)

        # Calculate geometric mean and save relevant statistics to data
        data = []
        for i in range(length):
            # Caclulate overall score as geometric mean of word_len_mean, vocab_norm
            # dep_depth_mean
            word_len_mean = self.texts_stats[names[i]]['word_len_mean']
            vocab_norm = self.texts_stats[names[i]]['vocab_norm']
            dep_depth_mean = self.texts_stats[names[i]]['dep_depth_mean']

            geometric_mean = (word_len_mean * vocab_norm *
                              dep_depth_mean)**(1. / 3)

            data.append([word_len_mean, vocab_norm,
                         dep_depth_mean, geometric_mean])
        data = np.asarray(data)

        #  Normalise data according to its category
        for j in range(data.shape[1]):
            max_value = np.max(data[:, j])
            data[:, j] = data[:, j] / max_value

        for i in range(data.shape[0]):
            ax.bar(x + (i * width), data[i, :], width, label=names[i])

        ax.set_ylabel('Average')
        # ax.set_ylim(0, 2)
        ax.set_xticks(x + width + width / 2)
        ax.set_xticklabels(x_labels)
        # ax.set_xlabel('Measure of Reading Difficulty')
        ax.set_title('Reading Difficulty of Different Texts')
        ax.legend()
        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

        # Shrink current axis's height by 20% on the bottom
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                  box.width, box.height * 2])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                  fancybox=True, shadow=True)

        fig.tight_layout()
        plt.savefig('./results/Bar Chart of Reading Difficulty.pdf')

    # This plots a line graph
    def plot_line(xs, ys, xlabel='x', ylabel='y', title='Title', label=None):
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if label is None:
            plt.plot(xs, ys, 'b')
        else:
            plt.plot(xs, ys, 'b', label=label)
        plt.legend(loc='upper right')
        plt.savefig('./plots/' + ylabel + '---' + xlabel + '.pdf')

    # Plots a double bar chart comparing the distribution of word lengths
    # between two different texts and save the bar chart to `plots`
    def plot_comparison_word_lengths(self, name1, name2):
        print('Plotting bar chart of word lengths in ' +
              name1 + ' and ' + name2 + '...')
        max_len = 15
        y1s = self.texts_dists[name1]['word_len'][0]
        x1s = self.texts_dists[name1]['word_len'][1]

        y2s = self.texts_dists[name2]['word_len'][0]
        x2s = self.texts_dists[name2]['word_len'][1]

        max_len = min([len(y1s), len(x1s), len(y2s), len(x2s)])

        self.plot_double_bar(x1s[:max_len], y1s[:max_len],
                             x2s[:max_len], y2s[:max_len],
                             w=0.4,
                             legend=[str(name1),
                                     str(name2)],
                             xlabel='Word Length (in characters)',
                             ylabel='Normalised Freqency',
                             title='Normalised Histogram of Word Length',
                             file_name='word-lengths-' + name1 + '-' + name2)

    # Plots a double bar chart comparing the distribution of dependency depths
    # between two different texts and save the bar chart to `plots`
    def plot_comparison_dependency_depths(self, name1, name2):
        print('Plotting bar chart of dependency depths in '
              + name1 + ' and ' + name2 + '...')
        y1s = self.texts_dists[name1]['dep_depth'][0]
        x1s = self.texts_dists[name1]['dep_depth'][1]

        y2s = self.texts_dists[name2]['dep_depth'][0]
        x2s = self.texts_dists[name2]['dep_depth'][1]

        max_len = min([len(y1s), len(x1s), len(y2s), len(x2s)])

        self.plot_double_bar(x1s[:max_len], y1s[:max_len],
                             x2s[:max_len], y2s[:max_len],
                             w=0.4,
                             legend=[str(name1),
                                     str(name2)],
                             xlabel='Dependency Depth of Word',
                             ylabel='Normalised Freqency',
                             title='Normalised Histogram of Word Dependency Depth',
                             file_name='word-dependency-depths-' + name1 + '-' + name2)

    # Plots a double bar chart comparing the distribution of sentence lengths
    # between two different texts and save the bar chart to `plots`
    def plot_comparison_sentence_lengths(self, name1, name2):
        print('Plotting bar chart of sentence lengths in '
              + name1 + ' and ' + name2 + '...')
        y1s = self.texts_dists[name1]['sent_len'][0]
        x1s = self.texts_dists[name1]['sent_len'][1]

        y2s = self.texts_dists[name2]['sent_len'][0]
        x2s = self.texts_dists[name2]['sent_len'][1]

        max_len = min([len(y1s), len(x1s), len(y2s), len(x2s), 25])

        self.plot_double_bar(x1s[:max_len], y1s[:max_len],
                             x2s[:max_len], y2s[:max_len],
                             w=0.4,
                             legend=[str(name1),
                                     str(name2)],
                             xlabel='Sentence Length',
                             ylabel='Normalised Freqency',
                             title='Normalised Histogram of Sentence Lengths',
                             file_name='sentence-lengths-' + name1 + '-' + name2)


if __name__ == "__main__":
    limit = 10000
    library = './sample_texts/'
    enable_gpu = False

    # Update settings according to arguments passed via the terminal
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-input':
            library = sys.argv[i + 1]
        elif sys.argv[i] == '-limit':
            limit = int(sys.argv[i + 1])
        elif sys.argv[i] == '-gpu':
            enable_gpu = sys.argv[i + 1].lower()
            if enable_gpu == 'true':
                enable_gpu = True
            elif enable_gpu == 'false':
                enable_gpu = False

    if library[-1] != '/':
        library += '/'

    print('========= TEXT ANALYZER =========')

    text_analyzer = Text_Analyzer(limit=limit, enable_gpu=enable_gpu)

    text_analyzer.load(library=library)

    text_analyzer.analyze()

    text_analyzer.export_statistics_CSV('./results/text_statistics.csv')

    filenames = glob.glob(library + '*.txt')
    filenames.sort()
    for i in range(len(filenames)):
        filenames[i] = filenames[i].replace(library, '')
        filenames[i] = filenames[i].replace('.txt', '')

    text_analyzer.plot_multi_bar(filenames)

    print('COMPLETED. See ./results for a bar graph and statistics.')
