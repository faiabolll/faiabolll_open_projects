--создание таблицы 1day CJM
with daterange as (
    select 
        '20210720' as datefrom_str,
        '20211130' as dateto_str,
        date('2021-07-20') as datefrom,
        date('2021-11-30') as dateto
),
events as (
    select
        hitId,
        anonymousId,
        userId,
        sentAt,
        context.page.url,
        event.name eventName,
        event.page.type eventPageType,
        event.page.category eventPageCategory,
        event.product.skuCode skuCode,
        event.product.categoryId productCategory,
        event.transaction.lineItems transactionCart,
        event.transaction.total transactionRevenue
    from `hits_*`
    where _table_suffix between (select datefrom_str from daterange) and (select dateto_str from daterange)
),
sessions as (
    select
        sessionId,
        context.campaign.source,
        context.campaign.medium,
        context.campaign.name campaign,
        hh.hitId hitId
    from `sessions_*`, unnest(hits) as hh
    where _table_suffix between (select datefrom_str from daterange) and (select dateto_str from daterange)
)
select * from events left join sessions using(hitId)