#Plotting a correlation matrix from the CSV genereated when testinng a model
library(tidyverse)

#Confusion matrix for EPI classifier tested on mixed dataset
conf_data <- read_csv("conf_matrix_REFORMAT.csv") #Data with ratio for predicted label on the actual real class (adds to 1 for all cells on same real class)

#Goal n differnt data format

HTMP_conf <- ggplot(conf_data, aes(real,predicted)) +
              geom_tile(aes(x=real,fill=Ratio)) +
              geom_text(aes(label=round(count,1))) +
              scale_fill_distiller(palette = "YlOrRd", direction = 1) +
              labs(title = "Type overlap plot",
                    subtitle = "(Counts are absolute cell numbers)") +
              theme_classic()
HTMP_conf

DotPlot_conf <- ggplot(conf_data, aes(real,predicted)) +
                      geom_point(shape=21, aes(x=real,
                                fill=Ratio, size=Ratio)) +
                      geom_label(nudge_y=-0.12, aes(label=round(count,1), 
                                alpha=0.6, fontface="bold")) +
                      scale_fill_distiller(palette = "YlOrRd", 
                                direction = 1, guide = "legend") +
                      scale_size(range=c(2,30)) +
                      scale_x_discrete(position = "bottom") +
                      labs(title = "EPI model performance against co-culture setting", 
                            subtitle = "(Counts are absolute cell numbers)",
                           x="Real state", y="Predicted state") +
                      guides(alpha=FALSE) +
                      theme_classic()
DotPlot_conf