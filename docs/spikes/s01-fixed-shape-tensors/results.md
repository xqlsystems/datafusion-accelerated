| target partitions | operator | runs | rows | tensor columns out | keeps its extension type | values |
|---|---|---|---|---|---|---|
| 1 | projection | yes | 1000 | 1 | yes | zero-copy |
| 1 | filter | yes | 500 | 1 | yes | copied |
| 1 | limit | yes | 10 | 1 | yes | zero-copy |
| 1 | sort | yes | 1000 | 1 | yes | copied |
| 1 | hash join, tensor as payload | yes | 334 | 1 | yes | copied |
| 1 | union all | yes | 2000 | 1 | yes | zero-copy |
| 1 | repartition by key | **no**: This feature is not implemented: Physical plan does not support DistributeBy partitioning | | | | |
| 1 | window | yes | 1000 | 1 | yes | copied |
| 1 | group by, first_value | yes | 10 | 1 | yes | copied |
| 1 | group by, array_agg | yes | 10 | 0 | n/a | n/a |
| 1 | element access | yes | 1000 | 0 | n/a | n/a |
| 1 | elementwise arithmetic | **no**: Error during planning: Cannot coerce arithmetic expression FixedSizeList(64 x non-null Float64) * Float64 to v | | | | |
| 1 | sum of tensors | **no**: Error during planning: Internal error: Function 'sum' failed to match any signature, errors: Error during plan | | | | |
| 1 | join on tensor equality | yes | 1000 | 0 | n/a | n/a |
| 1 | cast to list | yes | 1000 | 0 | n/a | n/a |
| 4 | projection | yes | 1000 | 1 | yes | zero-copy |
| 4 | filter | yes | 500 | 1 | yes | copied |
| 4 | limit | yes | 10 | 1 | yes | zero-copy |
| 4 | sort | yes | 1000 | 1 | yes | copied |
| 4 | hash join, tensor as payload | yes | 334 | 1 | yes | copied |
| 4 | union all | yes | 2000 | 1 | yes | zero-copy |
| 4 | repartition by key | **no**: This feature is not implemented: Physical plan does not support DistributeBy partitioning | | | | |
| 4 | window | yes | 1000 | 1 | yes | copied |
| 4 | group by, first_value | yes | 10 | 1 | yes | copied |
| 4 | group by, array_agg | yes | 10 | 0 | n/a | n/a |
| 4 | element access | yes | 1000 | 0 | n/a | n/a |
| 4 | elementwise arithmetic | **no**: Error during planning: Cannot coerce arithmetic expression FixedSizeList(64 x non-null Float64) * Float64 to v | | | | |
| 4 | sum of tensors | **no**: Error during planning: Internal error: Function 'sum' failed to match any signature, errors: Error during plan | | | | |
| 4 | join on tensor equality | yes | 1000 | 0 | n/a | n/a |
| 4 | cast to list | yes | 1000 | 0 | n/a | n/a |
