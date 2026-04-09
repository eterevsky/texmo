from texmo.common import is_power2, is_power2_int, power2_neighbors, itoa3, ttoa3


def test_itoa3():
    assert itoa3(0) == "0"
    assert itoa3(1) == "1"
    assert itoa3(900) == "900"
    assert itoa3(999) == "999"
    assert itoa3(1000) == "1000"
    assert itoa3(1800) == "1800"
    assert itoa3(1999) == "1999"
    assert itoa3(2000) == "2.00k"
    assert itoa3(2001) == "2.00k"
    assert itoa3(2010) == "2.01k"
    assert itoa3(12345) == "12.35k"
    assert itoa3(65536) == "65.5k"
    assert itoa3(123456) == "123.5k"
    assert itoa3(500000) == "500k"
    assert itoa3(1234567) == "1235k"
    assert itoa3(1999999) == "2.00M"
    assert itoa3(2345678) == "2.35M"


def test_is_power2():
    assert is_power2(1)
    assert is_power2(2)
    assert is_power2(1024)
    assert is_power2(0.5)
    assert is_power2(0.125)
    assert is_power2(16.0)
    assert not is_power2(0)
    assert not is_power2(-1)
    assert not is_power2(3)
    assert not is_power2(7)
    assert not is_power2(0.3)


def test_is_power2_int():
    assert is_power2_int(1)
    assert is_power2_int(64)
    assert not is_power2_int(0)
    assert not is_power2_int(-1)
    assert not is_power2_int(3)
    assert not is_power2_int(1.0)


def test_power2_neighbors():
    assert power2_neighbors(1) == (2,)
    assert power2_neighbors(2) == (1, 4)
    assert power2_neighbors(16) == (8, 32)


def test_ttoa3():
    assert ttoa3(None) == "null"
    assert ttoa3(0.5e-6) == "500 ns"
    assert ttoa3(0.005) == "5.00 ms"
    assert ttoa3(0.05) == "50.0 ms"
    assert ttoa3(0.5) == "500 ms"
    assert ttoa3(5) == "5.00 s"
    assert ttoa3(30) == "30.0 s"
    assert ttoa3(90) == "1 m 30 s"
    assert ttoa3(3700) == "1 h 2 m"
