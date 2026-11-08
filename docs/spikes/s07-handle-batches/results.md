## E1. A device segment at the leaf, under DataFusion's planner and default rules

| target partitions | source partitions | benefits from partitioning | maintains order | query | result | between device nodes | handle batch reaches | cross-device reads | leaked buffers |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | true | false | group by | matches |  |  | 0 | 0 |
| 1 | 1 | true | false | filter, sort, limit | matches |  |  | 0 | 0 |
| 1 | 1 | true | false | projection | matches |  |  | 0 | 0 |
| 1 | 1 | true | false | count | matches |  |  | 0 | 0 |
| 1 | 1 | true | false | self join | matches |  |  | 0 | 0 |
| 1 | 1 | true | false | window | matches |  |  | 0 | 0 |
| 1 | 1 | true | false | union | matches |  |  | 0 | 0 |
| 1 | 1 | true | true | group by | matches |  |  | 0 | 0 |
| 1 | 1 | true | true | filter, sort, limit | matches |  |  | 0 | 0 |
| 1 | 1 | true | true | projection | matches |  |  | 0 | 0 |
| 1 | 1 | true | true | count | matches |  |  | 0 | 0 |
| 1 | 1 | true | true | self join | matches |  |  | 0 | 0 |
| 1 | 1 | true | true | window | matches |  |  | 0 | 0 |
| 1 | 1 | true | true | union | matches |  |  | 0 | 0 |
| 1 | 1 | false | false | group by | matches |  |  | 0 | 0 |
| 1 | 1 | false | false | filter, sort, limit | matches |  |  | 0 | 0 |
| 1 | 1 | false | false | projection | matches |  |  | 0 | 0 |
| 1 | 1 | false | false | count | matches |  |  | 0 | 0 |
| 1 | 1 | false | false | self join | matches |  |  | 0 | 0 |
| 1 | 1 | false | false | window | matches |  |  | 0 | 0 |
| 1 | 1 | false | false | union | matches |  |  | 0 | 0 |
| 1 | 1 | false | true | group by | matches |  |  | 0 | 0 |
| 1 | 1 | false | true | filter, sort, limit | matches |  |  | 0 | 0 |
| 1 | 1 | false | true | projection | matches |  |  | 0 | 0 |
| 1 | 1 | false | true | count | matches |  |  | 0 | 0 |
| 1 | 1 | false | true | self join | matches |  |  | 0 | 0 |
| 1 | 1 | false | true | window | matches |  |  | 0 | 0 |
| 1 | 1 | false | true | union | matches |  |  | 0 | 0 |
| 1 | 3 | true | false | group by | matches |  |  | 0 | 0 |
| 1 | 3 | true | false | filter, sort, limit | matches |  |  | 0 | 0 |
| 1 | 3 | true | false | projection | matches |  |  | 0 | 0 |
| 1 | 3 | true | false | count | matches |  |  | 0 | 0 |
| 1 | 3 | true | false | self join | matches |  |  | 0 | 0 |
| 1 | 3 | true | false | window | matches |  |  | 0 | 0 |
| 1 | 3 | true | false | union | matches |  |  | 0 | 0 |
| 1 | 3 | true | true | group by | matches |  |  | 0 | 0 |
| 1 | 3 | true | true | filter, sort, limit | matches |  |  | 0 | 0 |
| 1 | 3 | true | true | projection | matches |  |  | 0 | 0 |
| 1 | 3 | true | true | count | matches |  |  | 0 | 0 |
| 1 | 3 | true | true | self join | matches |  |  | 0 | 0 |
| 1 | 3 | true | true | window | matches |  |  | 0 | 0 |
| 1 | 3 | true | true | union | matches |  |  | 0 | 0 |
| 1 | 3 | false | false | group by | matches |  |  | 0 | 0 |
| 1 | 3 | false | false | filter, sort, limit | matches |  |  | 0 | 0 |
| 1 | 3 | false | false | projection | matches |  |  | 0 | 0 |
| 1 | 3 | false | false | count | matches |  |  | 0 | 0 |
| 1 | 3 | false | false | self join | matches |  |  | 0 | 0 |
| 1 | 3 | false | false | window | matches |  |  | 0 | 0 |
| 1 | 3 | false | false | union | matches |  |  | 0 | 0 |
| 1 | 3 | false | true | group by | matches |  |  | 0 | 0 |
| 1 | 3 | false | true | filter, sort, limit | matches |  |  | 0 | 0 |
| 1 | 3 | false | true | projection | matches |  |  | 0 | 0 |
| 1 | 3 | false | true | count | matches |  |  | 0 | 0 |
| 1 | 3 | false | true | self join | matches |  |  | 0 | 0 |
| 1 | 3 | false | true | window | matches |  |  | 0 | 0 |
| 1 | 3 | false | true | union | matches |  |  | 0 | 0 |
| 4 | 1 | true | false | group by | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | false | filter, sort, limit | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | false | projection | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | false | count | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | false | self join | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 4 | 0 |
| 4 | 1 | true | false | window | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | false | union | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 4 | 0 |
| 4 | 1 | true | true | group by | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | true | filter, sort, limit | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | true | projection | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | true | count | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | true | self join | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 4 | 0 |
| 4 | 1 | true | true | window | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 2 | 0 |
| 4 | 1 | true | true | union | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=1 |  | 4 | 0 |
| 4 | 1 | false | false | group by | matches |  |  | 0 | 0 |
| 4 | 1 | false | false | filter, sort, limit | matches |  |  | 0 | 0 |
| 4 | 1 | false | false | projection | matches |  |  | 0 | 0 |
| 4 | 1 | false | false | count | matches |  |  | 0 | 0 |
| 4 | 1 | false | false | self join | matches |  |  | 0 | 0 |
| 4 | 1 | false | false | window | matches |  |  | 0 | 0 |
| 4 | 1 | false | false | union | matches |  |  | 0 | 0 |
| 4 | 1 | false | true | group by | matches |  |  | 0 | 0 |
| 4 | 1 | false | true | filter, sort, limit | matches |  |  | 0 | 0 |
| 4 | 1 | false | true | projection | matches |  |  | 0 | 0 |
| 4 | 1 | false | true | count | matches |  |  | 0 | 0 |
| 4 | 1 | false | true | self join | matches |  |  | 0 | 0 |
| 4 | 1 | false | true | window | matches |  |  | 0 | 0 |
| 4 | 1 | false | true | union | matches |  |  | 0 | 0 |
| 4 | 3 | true | false | group by | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | false | filter, sort, limit | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | false | projection | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | false | count | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | false | self join | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 12 | 0 |
| 4 | 3 | true | false | window | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | false | union | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 12 | 0 |
| 4 | 3 | true | true | group by | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | true | filter, sort, limit | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | true | projection | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | true | count | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | true | self join | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 12 | 0 |
| 4 | 3 | true | true | window | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 | 0 |
| 4 | 3 | true | true | union | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 12 | 0 |
| 4 | 3 | false | false | group by | matches |  |  | 0 | 0 |
| 4 | 3 | false | false | filter, sort, limit | matches |  |  | 0 | 0 |
| 4 | 3 | false | false | projection | matches |  |  | 0 | 0 |
| 4 | 3 | false | false | count | matches |  |  | 0 | 0 |
| 4 | 3 | false | false | self join | matches |  |  | 0 | 0 |
| 4 | 3 | false | false | window | matches |  |  | 0 | 0 |
| 4 | 3 | false | false | union | matches |  |  | 0 | 0 |
| 4 | 3 | false | true | group by | matches |  |  | 0 | 0 |
| 4 | 3 | false | true | filter, sort, limit | matches |  |  | 0 | 0 |
| 4 | 3 | false | true | projection | matches |  |  | 0 | 0 |
| 4 | 3 | false | true | count | matches |  |  | 0 | 0 |
| 4 | 3 | false | true | self join | matches |  |  | 0 | 0 |
| 4 | 3 | false | true | window | matches |  |  | 0 | 0 |
| 4 | 3 | false | true | union | matches |  |  | 0 | 0 |

## E2. A device segment placed mid-plan by a rule, first or last among DataFusion's rules

| rule position | target partitions | benefits from partitioning | query | result | between device nodes | handle batch reaches | cross-device reads |
|---|---|---|---|---|---|---|---|
| first | 1 | true | group by | matches |  |  | 0 |
| first | 1 | true | count | matches |  |  | 0 |
| first | 1 | true | self join | matches |  |  | 0 |
| first | 1 | false | group by | matches |  |  | 0 |
| first | 1 | false | count | matches |  |  | 0 |
| first | 1 | false | self join | matches |  |  | 0 |
| first | 4 | true | group by | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 |
| first | 4 | true | count | matches | RepartitionExec: partitioning=RoundRobinBatch(4), input_partitions=3 |  | 6 |
| first | 4 | true | self join | matches |  |  | 0 |
| first | 4 | false | group by | matches |  |  | 0 |
| first | 4 | false | count | matches |  |  | 0 |
| first | 4 | false | self join | matches |  |  | 0 |
| last | 1 | true | group by | matches |  |  | 0 |
| last | 1 | true | count | matches |  |  | 0 |
| last | 1 | true | self join | matches |  |  | 0 |
| last | 1 | false | group by | matches |  |  | 0 |
| last | 1 | false | count | matches |  |  | 0 |
| last | 1 | false | self join | matches |  |  | 0 |
| last | 4 | true | group by | matches |  |  | 0 |
| last | 4 | true | count | matches |  |  | 0 |
| last | 4 | true | self join | matches |  |  | 0 |
| last | 4 | false | group by | matches |  |  | 0 |
| last | 4 | false | count | matches |  |  | 0 |
| last | 4 | false | self join | matches |  |  | 0 |
