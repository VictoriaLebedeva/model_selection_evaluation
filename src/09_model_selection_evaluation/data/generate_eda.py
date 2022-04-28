import os
import click
import pandas as pd
import sweetviz as sv
from pandas_profiling import ProfileReport
from datetime import datetime


@click.command()
@click.option('--profiler', default='pandas-profiling', type=click.Choice(['pandas-profiling', 'sweetviz']), help="Profiler, which generates EDA report")
def generate_eda(profiler):
    df = pd.read_csv('..\\..\\data\\external\\train.csv')
    report_filename = f'report_{profiler}_{datetime.now.strftime("%d%m%Y_%H%M")}.html'
    output_path = os.path.join('..\\..\\data\\external\\eda_reports', report_filename)
    if profiler == 'pandas-profiling':
        profile = ProfileReport(df)
        profile.to_file(output_file=output_path)
    if profiler == 'sweetviz':
        sweet_report = sv.analyze(df)
        sweet_report.show_html(output_path) 

if __name__ == '__main__':
    generate_eda()

