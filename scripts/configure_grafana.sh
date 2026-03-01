#!/usr/bin/env bash
set -e
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
GRAFANA_USER="${GRAFANA_USER:-admin}"
GRAFANA_PASS="${GRAFANA_PASS:-admin}"
INFLUX_URL="${INFLUX_URL:-http://localhost:8086}"
INFLUX_ORG="${INFLUX_ORG:-jetson}"
INFLUX_BUCKET="${INFLUX_BUCKET:-person_detection}"
INFLUX_TOKEN="${INFLUX_TOKEN:-my-super-secret-auth-token}"
CREATE_PAYLOAD="$(cat <<EOF
{
  "name":"InfluxDB",
  "type":"influxdb",
  "access":"proxy",
  "url":"$INFLUX_URL",
  "isDefault":true,
  "jsonData":{
    "version":"Flux",
    "organization":"$INFLUX_ORG",
    "defaultBucket":"$INFLUX_BUCKET",
    "tlsSkipVerify":true
  },
  "secureJsonData":{
    "token":"$INFLUX_TOKEN"
  }
}
EOF
)"
code=$(curl -s -u "$GRAFANA_USER:$GRAFANA_PASS" -H "Content-Type: application/json" -X POST "$GRAFANA_URL/api/datasources" -d "$CREATE_PAYLOAD" -o /dev/null -w "%{http_code}")
if [ "$code" != "200" ] && [ "$code" != "202" ]; then
  ds_json=$(curl -s -u "$GRAFANA_USER:$GRAFANA_PASS" "$GRAFANA_URL/api/datasources")
  ds_id=$(python3 - <<PY
import json,sys
arr=json.loads(sys.stdin.read())
print(next((str(x["id"]) for x in arr if x.get("name")=="InfluxDB"), ""))
PY
<<< "$ds_json")
  [ -z "$ds_id" ] && exit 1
  UPDATE_PAYLOAD="$(cat <<EOF
{
  "id": $ds_id,
  "name":"InfluxDB",
  "type":"influxdb",
  "access":"proxy",
  "url":"$INFLUX_URL",
  "isDefault":true,
  "jsonData":{
    "version":"Flux",
    "organization":"$INFLUX_ORG",
    "defaultBucket":"$INFLUX_BUCKET",
    "tlsSkipVerify":true
  },
  "secureJsonFields":{},
  "secureJsonData":{
    "token":"$INFLUX_TOKEN"
  }
}
EOF
)"
  curl -s -u "$GRAFANA_USER:$GRAFANA_PASS" -H "Content-Type: application/json" -X PUT "$GRAFANA_URL/api/datasources/$ds_id" -d "$UPDATE_PAYLOAD" > /dev/null
fi
