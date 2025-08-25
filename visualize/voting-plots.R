library(ggplot2)
library(cowplot)
library(dplyr)
library(ggforce)
library(tidyr)


data_dir = '/Users/ethanhodess/Documents/Documents - Ethan’s MacBook Pro/Cedars/2025/TPOT_ensemble/collect/'
scores <- read.csv(paste(data_dir, 'combined_results.csv', sep = "", collapse = NULL), header = TRUE, stringsAsFactors = FALSE)


scores_long <- scores %>%
  pivot_longer(
    cols = c("individual", "model.1", "model.2", "model.3", "model.4", "model.5", "model.6", "model.7", "model.8"),  
    names_to = "model",
    values_to = "accuracy"
  )


model_accuracy = filter(scores_long, task.id == "359954") %>%
  ggplot(aes(x = model, y = accuracy, fill = model)) +
  geom_violin(trim = FALSE, alpha = 0.2) +
  geom_boxplot(width = 0.15, outlier.shape = NA, alpha = 0.0) +
  geom_jitter(aes(color = model), width = 0.15, alpha = 0.4) +
  theme_minimal() + 
  labs(title = "Voting Ensemble Accuracy vs Individual Model",
       x = "Model",
       y = "Accuracy") +
  theme(legend.position = "none", 
        plot.title = element_text(hjust = 0.5)) +
  scale_x_discrete(
    labels = c(
      "individual" = "Individual",
      "model.1" = "All, Hard",
      "model.2" = "All, Soft",
      "model.3" = "Top 50%, Hard",
      "model.4" = "Top 50%, Soft",
      "model.5" = "Random, Hard",
      "model.6" = "Random, Soft",
      "model.7" = "Weighted, Hard",
      "model.8" = "Weighted, Soft"
    )
  )

model_accuracy

