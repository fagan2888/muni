create table vehicle_monitoring (line_ref varchar(32),
				recorded_time timestamp,
                   valid_until_time timestamp,
                   direction_ref varchar(32),
                   data_frame_ref timestamp,
                   journey_ref int,
                   line_name varchar(1024),
                   operator_ref varchar(32),
                   monitored boolean,
                   vehicle_lat float,
                   vehicle_lon float,
                   vehicle_ref int,
                   stop_point_ref int,
                   visit_num int,
                   stop_point_name varchar(1024),
                   expected_arrival_time timestamp,
                   expected_departure_time timestamp)
                   distkey(line_ref)
					compound sortkey(line_ref,recorded_time);
