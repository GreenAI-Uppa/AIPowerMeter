import datetime
from deep_learning_power_measure.power_measure import experiment, parsers
from deep_learning_power_measure import labia


start = datetime.datetime(2023, 11, 7).timestamp()
end = datetime.datetime(2023, 11, 19).timestamp()

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
    summaries[user] = labia.summarize(user_stats) 
labia.print_summaries(per_status_summaries)
