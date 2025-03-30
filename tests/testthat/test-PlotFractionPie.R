test_that("plot_fraction_pie runs without error", {
    df <- data.frame(tumor = 0.055, normal = 0.945)
    expect_silent(PlotFractionPie(df))
})