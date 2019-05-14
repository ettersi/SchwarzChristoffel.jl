# https://github.com/JuliaLang/julia/issues/27613#issuecomment-400016111
function fargmax(f, itr)
    r = iterate(itr)
    r === nothing && error("empty collection")
    m, state = r
    f_m = f(m)
    while true
        r = iterate(itr, state)
        r === nothing && break
        x, state = r
        f_x = f(x)
        isless(f_m, f_x) || continue
        m, f_m = x, f_x
    end
    return m
end

function fargmin(f, itr)
    r = iterate(itr)
    r === nothing && error("empty collection")
    m, state = r
    f_m = f(m)
    while true
        r = iterate(itr, state)
        r === nothing && break
        x, state = r
        f_x = f(x)
        isless(f_x, f_m) || continue
        m, f_m = x, f_x
    end
    return m
end
