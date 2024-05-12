with t as (
    select 
        hitId,
        anonymousId,
        sentAt,
        if(eventName = 'Completed Transaction', 1, 0) as isTransaction
    from `1day CJM`
),
t1 as (
    select 
        *,
        sum(isTransaction) over (
            partition by anonymousId
            order by unix_millis(sentAt)
            rows between unbounded preceding and current row
        ) totalTransactions
    from t
)
select 
    *,
    count(hitId) over (
    partition by anonymousId, totalTransactions
    order by unix_millis(sentAt)
    range between 2592000000 preceding and current row
    ) eventNumBeforeTransaction
from t1
