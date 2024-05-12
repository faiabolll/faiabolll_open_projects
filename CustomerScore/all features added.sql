with t as (
    select 
        *,
        if(eventName = 'Completed Transaction', 1, 0) as isTransaction,
        first_value(sentAt) over (
            partition by sessionId
            order by sentAt asc
        ) session_start
    from `1day CJM`
),
t1 as (
    select 
        *,
        sum(isTransaction) over (
            partition by anonymousId
            order by unix_millis(sentAt)
            rows between unbounded preceding and current row
        ) totalTransactions_current,
        sum(isTransaction) over (
            partition by anonymousId
            rows between unbounded preceding and unbounded following 
        ) totalTransactionPerUser,
        dense_rank() over (
            partition by anonymousId
            order by unix_millis(session_start) asc
        ) session_num,
        max(session_start) over (
            partition by anonymousId
            order by unix_millis(session_start)
            range between unbounded preceding and 1000 preceding 
        ) previousSessionStart,
    from t
)
select 
    * except(transactionCart, hitId, userId),
    timestamp_diff(session_start, previousSessionStart, second) secondsSinceLastSession,
    timestamp_diff(sentAt, lag(sentAt) over (partition by anonymousId order by sentAt asc), second) secondsSinceLastEvent,
    count(hitId) over (
        partition by anonymousId, totalTransactions_current
        order by unix_millis(sentAt)
        range between 2592000000 preceding and current row
    ) eventNumBeforeTransaction,
    count(hitId) over (
        partition by anonymousId, sessionId
        order by unix_millis(sentAt)
    ) eventNumInSession,
    if(totalTransactionPerUser != 0, 1, 0) everTransacted 
from t1
limit 100000