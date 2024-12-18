import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc('font', family='arial')
mpl.rc('font', serif='arial')

# set run name
run_name = 'Denoising36'

F1_scores_path = f'results/{run_name}/F1 scores results - {run_name}.csv'
filenames = [f'F1 score for masking network prediction - {run_name}', f'F1 score for post process prediction - {run_name}']

# load df of the F1 scores table
F1_scores_df = pd.read_csv(F1_scores_path, index_col=0)
# Reorder the columns in the data frame by median
medians = F1_scores_df.median().sort_values(ascending=False)
F1_scores_df = F1_scores_df[medians.index]

# Create the box plot
num_channels = medians.shape[0]
plt.figure(figsize=(num_channels * 2 + 2, 30), dpi=300)
sns.set(style="white")
PROPS = {
'boxprops':{'facecolor':'grey', 'edgecolor':'black'},
'medianprops':{'color':'black'},
'whiskerprops':{'color':'black'},
'capprops':{'color':'black'}
}
ax = sns.boxplot(data=F1_scores_df, color='w', width=0.6, showfliers=False, linewidth=0.5, **PROPS)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(0.5)
ax.spines['left'].set_linewidth(0.5)

# Add median lines to the plot
medians = F1_scores_df.median().round(2)
for i, median in enumerate(medians):
    plt.text(i, median + 0.01, f"{median:.2f}", horizontalalignment='center', fontsize=24, color='black')

# Set the x-tick labels to channel names
plt.xticks(range(0, num_channels), F1_scores_df.columns, fontsize=30, rotation=90)
plt.yticks(fontsize=30)

# Add labels and title to the plot
plt.xlabel("\nProteins", fontsize=24)
plt.ylabel("F1 Scores\n", fontsize=24)
plt.ylim(0, 1)
plt.title("F1 Scores by protein\n", fontsize=30)

# Show the plot
plt.savefig(os.path.join(os.path.dirname(F1_scores_path), f'Box Plot - F1 scores - {run_name}.pdf'), format='pdf',bbox_inches='tight')
plt.show()


