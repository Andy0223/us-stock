from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


def ny_today(timezone: str = "America/New_York") -> date:
    return datetime.now(ZoneInfo(timezone)).date()


def is_us_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in nyse_holidays(day.year)


def is_us_early_close(day: date) -> bool:
    if not is_us_trading_day(day):
        return False
    thanksgiving = nth_weekday(day.year, 11, 3, 4)
    if day == thanksgiving + timedelta(days=1):
        return True
    if day.month == 12 and day.day == 24 and day.weekday() < 5:
        return True
    if day.month == 7 and day.day == 3 and day.weekday() < 5:
        return True
    return False


def previous_trading_day(day: date) -> date:
    cursor = day - timedelta(days=1)
    while not is_us_trading_day(cursor):
        cursor -= timedelta(days=1)
    return cursor


def next_trading_day(day: date) -> date:
    cursor = day + timedelta(days=1)
    while not is_us_trading_day(cursor):
        cursor += timedelta(days=1)
    return cursor


def nyse_holidays(year: int) -> set[date]:
    holidays = {
        observed_fixed(year, 1, 1),
        nth_weekday(year, 1, 0, 3),
        nth_weekday(year, 2, 0, 3),
        good_friday(year),
        last_weekday(year, 5, 0),
        observed_fixed(year, 6, 19),
        observed_fixed(year, 7, 4),
        nth_weekday(year, 9, 0, 1),
        nth_weekday(year, 11, 3, 4),
        observed_fixed(year, 12, 25),
    }
    return {item for item in holidays if item.year == year}


def observed_fixed(year: int, month: int, day: int) -> date:
    value = date(year, month, day)
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        cursor = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        cursor = date(year, month + 1, 1) - timedelta(days=1)
    while cursor.weekday() != weekday:
        cursor -= timedelta(days=1)
    return cursor


def good_friday(year: int) -> date:
    return easter_sunday(year) - timedelta(days=2)


def easter_sunday(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)
