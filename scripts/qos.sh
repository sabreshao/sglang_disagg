#!/bin/bash
# qos.sh

# Enable PFC and Auto-negotiation on all ports
for i in $(sudo nicctl show port | grep Port | awk {'print $3'}); do sudo nicctl update port -p $i --pause-type pfc --rx-pause enable --tx-pause enable; done
for i in $(sudo nicctl show port | grep Port | awk '{print $3}'); do sudo nicctl update port --port $i --auto-neg enable; done

# Define Priorities
cts_dscp=46
cts_prio=6
data_dscp=24
data_prio=3
default_prio=0
cnp_dscp=46
cnp_prio=6

sudo nicctl update qos pfc --priority 0 --no-drop disable
sudo nicctl update qos dscp-to-purpose --dscp 48 --purpose none
sudo nicctl update qos dscp-to-purpose --dscp 46 --purpose none
sudo nicctl update qos --classification-type pcp
sudo nicctl update qos --classification-type dscp
sudo nicctl update qos dscp-to-priority --dscp 0-63 --priority 0
sudo nicctl update qos dscp-to-priority --dscp 0-23,25-45,47-63 --priority $default_prio
sudo nicctl update qos dscp-to-priority --dscp $cts_dscp --priority $cts_prio
sudo nicctl update qos dscp-to-priority --dscp $data_dscp --priority $data_prio
sudo nicctl update qos dscp-to-priority --dscp $cnp_dscp --priority $cnp_prio
sudo nicctl update qos pfc --priority $data_prio --no-drop enable
sudo nicctl update qos scheduling --priority $data_prio,$default_prio,$cts_prio --dwrr 99,1,0 --rate-limit 0,0,10
