import os
import click
import pandas as pd
import sweetviz as sv
from pandas_profiling import ProfileReport
from datetime import datetime


@click.command()
@click.option(
    "--profiler",
    default="pandas-profiling",
    type=click.Choice(["pandas-profiling", "sweetviz"]),
    help="Profiler, which generates EDA report",
)
@click.option(
    "-d",
    "--dataset-path",
    default="data/external/train.csv",
    type=click.Path(exists=True, dir_okay=False),
    show_default=True,
)
@click.option(
    "-rp",
    "--report-path",
    default="data/eda_reports",
    type=click.Path(exists=True, dir_okay=True),
    show_default=True,
)
def generate_eda(profiler, dataset_path, report_path):
    """
    Generates EDA report and saves to the given directory.

    """
    df = pd.read_csv(dataset_path)
    now = datetime.now()
    report_filename = f'report_{profiler}_{now.strftime("%d%m%Y_%H%M")}.html'
    output_path = os.path.join(report_path, report_filename)
    if profiler == "pandas-profiling":
        profile = ProfileReport(df)
        profile.to_file(output_file=output_path)
    if profiler == "sweetviz":
        sweet_report = sv.analyze(df)
        sweet_report.show_html(output_path)
    click.echo(f"Report was created in {output_path}")


if __name__ == "__main__":
    generate_eda()
