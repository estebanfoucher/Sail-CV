# Jetson Nano

## Network

Setup: Mac => router => cameras + Jetson Nano

The Mac must be on the `192.168.1.x` subnet (via the router) to reach the
cameras and Jetson. When wired correctly the Mac gets a DHCP lease like
`192.168.1.50`. A self-assigned `169.254.x.x` address means the ethernet is
plugged in but not getting DHCP from the router.

### Devices

| Device      | IP:Port           | Service | Notes                        |
|-------------|-------------------|---------|------------------------------|
| Jetson Nano | `192.168.1.100:22`| SSH     | `ssh estebanfoucher@192.168.1.100` (pwd: eaglesailvision) |
| Cam1        | `192.168.1.34:554`| RTSP    | `rtsp://192.168.1.34:554/stream1`  |
| Cam2        | `192.168.1.214:554`| RTSP   | `rtsp://192.168.1.214:554/stream1` |

Camera RTSP URLs are configured in `viewer/reconstruct_loop.py` (`CAM1_URL` /
`CAM2_URL`, overridable via env vars).

### Verifying connectivity

```sh
# Confirm the Mac is on the router subnet
ifconfig | grep "inet 192.168"

# Ping each device
for ip in 192.168.1.100 192.168.1.34 192.168.1.214; do ping -c 2 $ip; done

# Confirm service ports are open
nc -z 192.168.1.34 554    # Cam1 RTSP
nc -z 192.168.1.214 554   # Cam2 RTSP
nc -z 192.168.1.100 22    # Jetson SSH
```

Last verified wired and reachable: 2026-05-28 (all devices ping ~1ms, RTSP and
SSH ports open).
