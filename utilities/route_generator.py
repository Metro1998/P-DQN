# @author Metro
# @date 2021/12/17


import random


def generate_routefile(seed, demand: list):
    """
    Generate XXX.rou.xml which will generate route in sumo respectively.

    :param demand: The generation possibility for each route(represented in veh/s)
                   [ES, EW, NE, NS, WN, WE, SW, SN, EN, NW, WS, SE]
                   e.g.
                   [[1./12, 1./13, 1./12, 1./7, 1./9, 1./10, 1./12, 1./7],
                                              ...
                    [1.13/, 1./14, 1./19, 1./5, 1./8, 1./15, 1./15, 1./19]]
                   len(demand) is chosen by yourself which decides the granularity of traffic situation.



    :param seed:
    :return:
    """
    assert isinstance(demand, list), 'Wrong data structure, a list of lists is required.'
    assert isinstance(demand[0], list), 'Wrong data structure, a list of lists is required.'

    random.seed(seed)
    N = 1800  # number of time steps for one simulation
    assert N % len(demand) == 0, 'N should be divisible by len(demand).'

    with open('envs/sumo/road_network/FW_Inter.rou.xml', "w") as routes:
        print("""<routes>
    <vType id="NE_left" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="private" speedFactor="norm(0.9, 0.15)"
           color="128, 128, 128" />
    <vType id="ES_left" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="private" speedFactor="norm(0.9, 0.15)"
           color="128, 128, 128" />
    <vType id="SW_left" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="private" speedFactor="norm(0.9, 0.15)"
           color="128, 128, 128"/>
    <vType id="WN_left" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="private" speedFactor="norm(0.9, 0.15)"
           color="128, 128, 128"/>

    <vType id="NS_through" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="custom1" speedFactor="norm(0.9, 0.15)"
           color="255, 255, 0"/>
    <vType id="EW_through" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="custom1" speedFactor="norm(0.9, 0.15)"
           color="255, 255, 0"/>
    <vType id="SN_through" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="custom1" speedFactor="norm(0.9, 0.15)"
           color="255, 255, 0"/>
    <vType id="WE_through" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="custom1" speedFactor="norm(0.9, 0.15)"
           color="255, 255, 0"/>

    <vType id="NW_right" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="custom2" speedFactor="norm(0.9, 0.15)"
           color="128, 255, 255"/>
    <vType id="EN_right" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="custom2" speedFactor="norm(0.9, 0.15)"
           color="128, 255, 255"/>
    <vType id="SE_right" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="custom2" speedFactor="norm(0.9, 0.15)"
           color="128, 255, 255"/>
    <vType id="WS_right" accel="3.0" decel="4.5" sigma="0.5" length="5" vClass="custom2" speedFactor="norm(0.9, 0.15)"
           color="128, 255, 255"/>

    <route id="EN" edges="east_in north_out" />
    <route id="EW" edges="east_in west_out" />
    <route id="ES" edges="east_in south_out" />
    <route id="NW" edges="north_in west_out" />
    <route id="NS" edges="north_in south_out" />
    <route id="NE" edges="north_in east_out" />
    <route id="WS" edges="west_in south_out" />
    <route id="WE" edges="west_in east_out" />
    <route id="WN" edges="west_in north_out" />
    <route id="SE" edges="south_in east_out" />
    <route id="SN" edges="south_in north_out" />
    <route id="SW" edges="south_in west_out" />
        """, file=routes)
        vehicle_Nr = 0
        for i in range(len(demand)):
            pES, pEW, pNE, pNS, pWN, pWE, pSW, pSN, pEN, pNW, pWS, pSE = demand[i]

            for j in range(int(N / len(demand))):

                if random.uniform(0, 1) < pES:
                    print('    <vehicle id="%i" type="ES_left" route="ES" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pEW:
                    print('    <vehicle id="%i" type="EW_through" route="EW" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pNE:
                    print('    <vehicle id="%i" type="NE_left" route="NE" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pNS:
                    print('    <vehicle id="%i" type="NS_through" route="NS" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pWN:
                    print('    <vehicle id="%i" type="WN_left" route="WN" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pWE:
                    print('    <vehicle id="%i" type="WE_through" route="WE" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pSW:
                    print('    <vehicle id="%i" type="SW_left" route="SW" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pSN:
                    print('    <vehicle id="%i" type="SN_through" route="SN" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pEN:
                    print('    <vehicle id="%i" type="EN_right" route="EN" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pNW:
                    print('    <vehicle id="%i" type="NW_right" route="NW" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pWS:
                    print('    <vehicle id="%i" type="WS_right" route="WS" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
                if random.uniform(0, 1) < pSE:
                    print('    <vehicle id="%i" type="SE_right" route="SE" depart="%i" departSpeed="random" />' % (
                        vehicle_Nr, j + int(N / len(demand)) * i), file=routes)
                    vehicle_Nr += 1
        print("</routes>", file=routes)











