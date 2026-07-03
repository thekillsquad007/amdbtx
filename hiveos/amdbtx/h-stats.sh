#!/usr/bin/env bash

LOG_BASENAME="${CUSTOM_LOG_BASENAME:-/var/log/miner/amdbtx/lastrun}"
LOG_FILE="${LOG_BASENAME}.log"
START_FILE="/var/run/amdbtx.started"
MINER_VERSION="${CUSTOM_VERSION:-1.2.0_hiveos}"

json_num_array() {
    local raw="$1"
    local out="" item
    raw="${raw//,/ }"
    for item in $raw; do
        [[ "$item" =~ ^-?[0-9]+([.][0-9]+)?$ ]] || continue
        out="${out:+$out,}$item"
    done
    printf '[%s]' "$out"
}

last_solve_line() {
    [[ -f "$LOG_FILE" ]] || return 0
    tail -n 300 "$LOG_FILE" 2>/dev/null | grep 'solve:' | tail -n 1 || true
}

line="$(last_solve_line)"
khs="0"
hs_json="[0]"
accepted="0"
rejected="0"

if [[ -n "$line" ]]; then
    khs="$(sed -nE 's/.*nonce_khps=([0-9.]+).*/\1/p' <<<"$line")"
    [[ -n "$khs" ]] || khs="0"

    per_gpu="$(sed -nE 's/.*nonce_khps=[0-9.]+ total \(([^)]*) per GPU\).*/\1/p' <<<"$line")"
    if [[ -n "$per_gpu" ]]; then
        hs_json="$(json_num_array "${per_gpu//+/ }")"
    else
        hs_json="[$khs]"
    fi

    accepted="$(sed -nE 's/.*shares=([0-9]+)\/([0-9]+).*/\1/p' <<<"$line")"
    rejected="$(sed -nE 's/.*shares=([0-9]+)\/([0-9]+).*/\2/p' <<<"$line")"
    [[ -n "$accepted" ]] || accepted="0"
    [[ -n "$rejected" ]] || rejected="0"
fi

now="$(date +%s)"
started="$(cat "$START_FILE" 2>/dev/null || echo "$now")"
[[ "$started" =~ ^[0-9]+$ ]] || started="$now"
uptime=$((now - started))
[[ "$uptime" -ge 0 ]] || uptime=0

temp_json="$(json_num_array "${gpu_temp:-${GPU_TEMP:-${temps:-}}}")"
fan_json="$(json_num_array "${gpu_fan:-${GPU_FAN:-${fans:-}}}")"
bus_json="$(json_num_array "${bus_numbers:-${BUS_NUMBERS:-}}")"

stats=$(
    printf '{"hs":%s,"hs_units":"khs","temp":%s,"fan":%s,"bus_numbers":%s,"ar":[%s,%s],"uptime":%s,"ver":"%s"}' \
        "$hs_json" "$temp_json" "$fan_json" "$bus_json" "$accepted" "$rejected" "$uptime" "$MINER_VERSION"
)
