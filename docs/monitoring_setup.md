# Monitoring Setup (Jetson)

- InfluxDB
  - Create org: POST /api/v2/orgs {"name":"jetson"} with Authorization: Token
  - Find orgId: GET /api/v2/orgs?name=jetson
  - Create bucket: POST /api/v2/buckets {"orgID":"<orgId>","name":"person_detection","retentionRules":[{"type":"expire","everySeconds":0}]}
  - Create token: POST /api/v2/authorizations {"orgID":"<orgId>","description":"yolo11-token","permissions":[{"action":"read","resource":{"type":"buckets","orgID":"<orgId>"}},{"action":"write","resource":{"type":"buckets","orgID":"<orgId>"}}]}
  - Use INFLUX_URL=http://localhost:8086, INFLUX_ORG=jetson, INFLUX_BUCKET=person_detection, INFLUX_TOKEN=<generated>
- Grafana
  - Configure datasource with scripts/configure_grafana.sh
  - Import dashboard: POST /api/dashboards/import with grafana/dashboards/yolo11n_dashboard.json
  - Access dashboard: /d/yolo11n-dashboard/yolo11n-person-detection-dashboard
