br = Baroreflex()
assert br.get_state_size() == 3
y0 = br.get_initial_state()
assert np.allclose(y0, [70.0, 1.0, 1.0])

# При P_sa = P_set производные должны быть нулевыми
dy = br.get_derivatives(0.0, y0, {'P_sa': 80.0, 'P_pa': 15.0})
assert np.allclose(dy, [0.0, 0.0, 0.0], atol=1e-12)

# При P_sa = 100 (гипертензия) все три ветви идут вниз
dy = br.get_derivatives(0.0, y0, {'P_sa': 100.0})
assert dy[0] < 0     # HR падает
assert dy[1] < 0     # инотропия падает
assert dy[2] < 0     # вазомотор падает

# При P_sa = 60 (гипотензия) все три растут
dy = br.get_derivatives(0.0, y0, {'P_sa': 60.0})
assert dy[0] > 0 and dy[1] > 0 and dy[2] > 0

# Legacy-совместимость
br2 = Baroreflex(gain=0.015, tau=1.5)
assert br2.k_hr == 0.015
assert br2.tau_hr == 1.5

# Старый k_inotropy=1.5 должен упасть
try:
    Baroreflex(k_inotropy=1.5)
    assert False, "должно было упасть на валидации"
except ValueError as e:
    assert "k_inotropy" in str(e)