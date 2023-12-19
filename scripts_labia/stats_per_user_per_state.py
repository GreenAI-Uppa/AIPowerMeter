import datetime
from deep_learning_power_measure import labia
import argparse

parser = argparse.ArgumentParser("Display statistics for each user and each slurm job state")
parser.add_argument('--start',help='start date in isoformat',default='2023-11-16')
parser.add_argument('--end',help='end date in isoformat',default='2023-11-17')
args = parser.parse_args()


start = datetime.datetime.fromisoformat(args.start).timestamp()
end = datetime.datetime.fromisoformat(args.end).timestamp()

## display per user
all_user_stats = labia.get_stats_all_users(start, end)
summaries = {}
for user, user_stats in all_user_stats.items():
    summaries[user] = labia.summarize(user_stats) 
labia.print_summaries(summaries)

# display per status
per_status = labia.get_stats_per_job_status(all_user_stats)
per_status_summaries = {}
for user, user_stats in per_status.items():
    per_status_summaries[user] = labia.summarize(user_stats) 
labia.print_summaries(per_status_summaries)
