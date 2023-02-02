# Author: Roberto Doriguzzi-Corin
# Project: Lecture on Intrusion Detection with Deep Learning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import pyshark
import socket
import random
import hashlib
import ipaddress
from sklearn.feature_extraction.text import CountVectorizer
from util_functions import *

IDS2018_DDOS_FLOWS = {'attackers': ['18.218.115.60', '18.219.9.1','18.219.32.43','18.218.55.126','52.14.136.135','18.219.5.43','18.216.200.189','18.218.229.235','18.218.11.51','18.216.24.42'],
                      'victims': ['18.218.83.150','172.31.69.28']}

IDS2017_DDOS_FLOWS = {'attackers': ['172.16.0.1'],
                      'victims': ['192.168.10.50']}

CUSTOM_DDOS_SYN = {'attackers': ['11.0.0.' + str(x) for x in range(1,255)],
                      'victims': ['10.42.0.2']}

DOS2019_FLOWS = {'attackers': ['172.16.0.5'], 'victims': ['192.168.50.1', '192.168.50.4']}

DDOS_ATTACK_SPECS = {
    'DOS2017' : IDS2017_DDOS_FLOWS,
    'DOS2018' : IDS2018_DDOS_FLOWS,
    'SYN2020' : CUSTOM_DDOS_SYN,
    'DOS2019': DOS2019_FLOWS
}


vector_proto = CountVectorizer()
vector_proto.fit_transform(protocols).todense()

random.seed(SEED)
np.random.seed(SEED)

class packet_features:
    def __init__(self):
        self.id_fwd = (0,0,0,0,0) # 5-tuple src_ip_addr, src_port,,dst_ip_addr,dst_port,protocol
        self.id_bwd = (0,0,0,0,0)  # 5-tuple src_ip_addr, src_port,,dst_ip_addr,dst_port,protocol
        self.features_list = []


    def __str__(self):
        return "{} -> {}".format(self.id_fwd, self.features_list)

def get_ddos_flows(attackers,victims):
    DDOS_FLOWS = {}

    if '/' in attackers: # subnet
        DDOS_FLOWS['attackers'] = [str(ip) for ip in list(ipaddress.IPv4Network(attackers).hosts())]
    else: # single address
        DDOS_FLOWS['attackers'] = [str(ipaddress.IPv4Address(attackers))]

    if '/' in victims:  # subnet
        DDOS_FLOWS['victims'] = [str(ip) for ip in list(ipaddress.IPv4Network(victims).hosts())]
    else:  # single address
        DDOS_FLOWS['victims'] = [str(ipaddress.IPv4Address(victims))]

    return DDOS_FLOWS

# function that build the labels based on the dataset type
def parse_labels(dataset_type=None, attackers=None,victims=None, label=1):
    output_dict = {}

    if attackers is not None and victims is not None:
        DDOS_FLOWS = get_ddos_flows(attackers, victims)
    elif dataset_type is not None and dataset_type in DDOS_ATTACK_SPECS:
        DDOS_FLOWS = DDOS_ATTACK_SPECS[dataset_type]
    else:
        return None

    for attacker in DDOS_FLOWS['attackers']:
        for victim in DDOS_FLOWS['victims']:
            ip_src = str(attacker)
            ip_dst = str(victim)
            key_fwd = (ip_src, ip_dst)
            key_bwd = (ip_dst, ip_src)

            if key_fwd not in output_dict:
                output_dict[key_fwd] = label
            if key_bwd not in output_dict:
                output_dict[key_bwd] = label

    return output_dict

def parse_packet(pkt):
    pf = packet_features()
    tmp_id = [0,0,0,0,0]

    try:
        pf.features_list.append(float(pkt.sniff_timestamp))  # timestampchild.find('Tag').text
        pf.features_list.append(int(pkt.ip.len))  # packet length
        pf.features_list.append(int(hashlib.sha256(str(pkt.highest_layer).encode('utf-8')).hexdigest(),
                                    16) % 10 ** 8)  # highest layer in the packet
        pf.features_list.append(int(int(pkt.ip.flags, 16)))  # IP flags
        tmp_id[0] = str(pkt.ip.src)  # int(ipaddress.IPv4Address(pkt.ip.src))
        tmp_id[2] = str(pkt.ip.dst)  # int(ipaddress.IPv4Address(pkt.ip.dst))

        protocols = vector_proto.transform([pkt.frame_info.protocols]).toarray().tolist()[0]
        protocols = [1 if i >= 1 else 0 for i in
                     protocols]  # we do not want the protocols counted more than once (sometimes they are listed twice in pkt.frame_info.protocols)
        protocols_value = int(np.dot(np.array(protocols), powers_of_two))
        pf.features_list.append(protocols_value)

        protocol = int(pkt.ip.proto)
        tmp_id[4] = protocol
        if pkt.transport_layer != None:
            if protocol == socket.IPPROTO_TCP:
                tmp_id[1] = int(pkt.tcp.srcport)
                tmp_id[3] = int(pkt.tcp.dstport)
                pf.features_list.append(int(pkt.tcp.len))  # TCP length
                pf.features_list.append(int(pkt.tcp.ack))  # TCP ack
                pf.features_list.append(int(pkt.tcp.flags, 16))  # TCP flags
                pf.features_list.append(int(pkt.tcp.window_size_value))  # TCP window size
                pf.features_list = pf.features_list + [0, 0]  # UDP + ICMP positions
            elif protocol == socket.IPPROTO_UDP:
                pf.features_list = pf.features_list + [0, 0, 0, 0]  # TCP positions
                tmp_id[1] = int(pkt.udp.srcport)
                pf.features_list.append(int(pkt.udp.length))  # UDP length
                tmp_id[3] = int(pkt.udp.dstport)
                pf.features_list = pf.features_list + [0]  # ICMP position
        elif protocol == socket.IPPROTO_ICMP:
            pf.features_list = pf.features_list + [0, 0, 0, 0, 0]  # TCP and UDP positions
            pf.features_list.append(int(pkt.icmp.type))  # ICMP type
        else:
            pf.features_list = pf.features_list + [0, 0, 0, 0, 0, 0]  # padding for layer3-only packets
            tmp_id[4] = 0

        pf.id_fwd = (tmp_id[0], tmp_id[1], tmp_id[2], tmp_id[3], tmp_id[4])
        pf.id_bwd = (tmp_id[2], tmp_id[3], tmp_id[0], tmp_id[1], tmp_id[4])

        return pf

    except AttributeError as e:
        # ignore packets that aren't TCP/UDP or IPv4
        return None

# Transforms live traffic into input samples for inference
def process_live_traffic(cap, in_labels, max_flow_len, traffic_type='all',time_window=TIME_WINDOW):
    start_time = time.time()
    temp_dict = OrderedDict()
    labelled_flows = []

    start_time_window = start_time
    time_window = start_time_window + time_window

    if isinstance(cap, pyshark.LiveCapture) == True:
        for pkt in cap.sniff_continuously():
            if time.time() >= time_window:
                break
            pf = parse_packet(pkt)
            temp_dict = store_packet(pf, temp_dict, start_time_window, max_flow_len)
    elif isinstance(cap, pyshark.FileCapture) == True:
        while time.time() < time_window:
            pkt = cap.next()
            pf = parse_packet(pkt)
            temp_dict = store_packet(pf,temp_dict,start_time_window,max_flow_len)

    apply_labels(temp_dict,labelled_flows, in_labels,traffic_type)
    return labelled_flows

def store_packet(pf,temp_dict,start_time_window, max_flow_len):
    if pf is not None:
        if pf.id_fwd in temp_dict and start_time_window in temp_dict[pf.id_fwd] and \
                temp_dict[pf.id_fwd][start_time_window].shape[0] < max_flow_len:
            temp_dict[pf.id_fwd][start_time_window] = np.vstack(
                [temp_dict[pf.id_fwd][start_time_window], pf.features_list])
        elif pf.id_bwd in temp_dict and start_time_window in temp_dict[pf.id_bwd] and \
                temp_dict[pf.id_bwd][start_time_window].shape[0] < max_flow_len:
            temp_dict[pf.id_bwd][start_time_window] = np.vstack(
                [temp_dict[pf.id_bwd][start_time_window], pf.features_list])
        else:
            if pf.id_fwd not in temp_dict and pf.id_bwd not in temp_dict:
                temp_dict[pf.id_fwd] = {start_time_window: np.array([pf.features_list]), 'label': 0}
            elif pf.id_fwd in temp_dict and start_time_window not in temp_dict[pf.id_fwd]:
                temp_dict[pf.id_fwd][start_time_window] = np.array([pf.features_list])
            elif pf.id_bwd in temp_dict and start_time_window not in temp_dict[pf.id_bwd]:
                temp_dict[pf.id_bwd][start_time_window] = np.array([pf.features_list])
    return temp_dict

def apply_labels(flows, labelled_flows, labels, traffic_type):
    for five_tuple, flow in flows.items():
        if labels is not None:
            short_key = (five_tuple[0], five_tuple[2])  # for IDS2017/IDS2018 dataset the labels have shorter keys
            flow['label'] = labels.get(short_key, 0)

        for flow_key, packet_list in flow.items():
            # relative time wrt the time of the first packet in the flow
            if flow_key != 'label':
                amin = np.amin(packet_list,axis=0)[0]
                packet_list[:, 0] = packet_list[:, 0] - amin

        if traffic_type == 'ddos' and flow['label'] == 0: # we only want malicious flows from this dataset
            continue
        elif traffic_type == 'benign' and flow['label'] > 0: # we only want benign flows from this dataset
            continue
        else:
            labelled_flows.append((five_tuple,flow))

# convert the dataset from dictionaries with 5-tuples keys into a list of flow fragments and another list of labels
def dataset_to_list_of_fragments(dataset):
    keys = []
    X = []
    y = []

    for flow in dataset:
        tuple = flow[0]
        flow_data = flow[1]
        label = flow_data['label']
        for key, fragment in flow_data.items():
            if key != 'label':
                X.append(fragment)
                y.append(label)
                keys.append(tuple)

    return X,y,keys

