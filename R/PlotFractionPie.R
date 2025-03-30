#' @title
#' Plot Pie Chart
#'
#' @description
#' Generate a pie chart for a vector of class fractions (e.g., tissue 
#' composition or cfDNA fractions). Automatically filters small values 
#' into an "Other" group, and allows for custom colors and font size control.
#'
#' @param fraction_vector a named numeric vector or one-row data.frame, 
#' where each value represents a class proportion.
#' @param title the title of the plot.
#' @param threshold a numeric value. Classes with fraction values below this 
#' threshold will be grouped into "Other".
#' @param class_colors a named character vector assigning colors to specific 
#' class names (e.g., c("tumor" = "red")).
#' @param font_size numeric, font scaling factor (default is 1.0).
#'
#' @return A pie chart is plotted to the current device.
#'
#' @examples
#' df <- data.frame(
#'   WBC = 0.93,
#'   Liver = 0.04,
#'   Lung = 0.02,
#'   Muscle = 1.2345e-4,
#'   Stomach = 9.87655e-03
#' )
#' PlotFractionPie(df, title = "cfDNA Composition", font_size = 1.2)
#'
#' @export
PlotFractionPie <- function(fraction_vector,
                              title = "Composition",
                              threshold = 0.01,
                              class_colors = NULL,
                              font_size = 1.0) {
    # Convert one-row data frame to named numeric vector
    if (is.data.frame(fraction_vector) && nrow(fraction_vector) == 1) {
        class_names <- colnames(fraction_vector)
        fraction_vector <- as.numeric(fraction_vector)
        names(fraction_vector) <- class_names
    }
    
    # Sort and filter
    fraction_vector <- sort(fraction_vector, decreasing = TRUE)
    is_other <- fraction_vector < threshold
    other_components <- fraction_vector[is_other]
    filtered <- fraction_vector[!is_other]
    
    if (length(other_components) > 0) {
        filtered <- c(filtered, Other = sum(other_components))
    }
    
    # Percentages
    percent <- filtered * 100
    slice_labels <- paste0(round(percent, 2), "%")
    
    # Format percentages for legend
    format_percent <- function(x) {
        if (x >= 0.001) {
            paste0(format(round(x, 3), nsmall = 3), "%")
        } else {
            paste0(formatC(x, format = "e", digits = 3), "%")
        }
    }
    
    # Legend labels
    legend_labels <- paste0(names(filtered), " (", vapply(percent, format_percent, character(1)), ")")
    
    if ("Other" %in% names(filtered)) {
        other_percents <- other_components * 100
        other_breakdown <- paste0(
            "   - ", names(other_components),
            " (", vapply(other_percents, format_percent, character(1)), ")"
        )
        legend_labels <- append(legend_labels, c("Other includes:", other_breakdown),
                                after = which(names(filtered) == "Other"))
    }
    
    # Assign colors
    all_class_names <- names(filtered)
    default_colors <- rainbow(length(filtered))
    colors <- setNames(default_colors, all_class_names)
    
    if (!is.null(class_colors)) {
        matched <- intersect(names(class_colors), names(filtered))
        colors[matched] <- class_colors[matched]
    }
    
    # Layout and margins — bring title and legend closer
    old_par <- par(no.readonly = TRUE)
    on.exit(par(old_par))
    
    # Adjust layout: smaller gap between pie and legend
    layout(matrix(1:2, nrow = 1), widths = c(2, 1.5))
    
    # Pie margins: reduce top margin to bring title closer
    par(mar = c(2, 1, 2, 1))  # bottom, left, top, right
    
    pie(filtered,
        labels = slice_labels,
        col = colors,
        main = title,
        cex.main = font_size * 1.2,
        cex = font_size)
    
    # Legend margins: shrink all sides slightly
    par(mar = c(1, 0, 2, 1))  # less padding for legend area
    plot.new()
    legend("left",
           legend = legend_labels,
           fill = c(colors, rep(NA, length(legend_labels) - length(colors))),
           border = NA,
           bty = "n",
           cex = font_size * 0.9,
           x.intersp = 0.5,
           y.intersp = 0.9)  # slightly reduce vertical spacing
}