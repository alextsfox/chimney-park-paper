def read_fullout(d):
    """collects full_output files from a directory"""
    all_data = []
    i = 0
    for run_dir in list(d.iterdir()):
        site = '-'.join(run_dir.name.split('-')[:-1])
        try:
            read_fullout_kwargs = dict(skiprows=[0, 2], parse_dates=[['date', 'time']], na_values=-9999)
            fullout = pd.concat([pd.read_csv(f, **read_fullout_kwargs) for f in run_dir.glob('*full_output*.csv')])
            fullout = fullout.loc[~fullout.date_time.duplicated()]
            
            date_time = pd.date_range(fullout.date_time.min(), fullout.date_time.max(), freq='30min')
            new_time = pd.DataFrame(dict(date_time=date_time))
            
            fullout = (
                fullout
                .merge(new_time, how='right', on='date_time')
                .sort_values('date_time')
                .set_index('date_time')
            )
            fullout['site'] = site
            fullout['date'] = fullout.index.date
            fullout['hour'] = fullout.index.hour + fullout.index.minute/60

            all_data.append(fullout)

        except ValueError:
            pass
        if i % 100 == 0: print(i)
        i += 1
    all_data = pd.concat(all_data)
    return all_data

def read_fluxnet(d):
    """collects fluxnet files from a directory"""
    all_data = []
    i = 0
    for run_dir in list(d.iterdir()):
        try:
            site = '-'.join(run_dir.name.split('-')[:-1])
            read_fullout_kwargs = dict(na_values=-9999)
            fullout = pd.concat([pd.read_csv(f, **read_fullout_kwargs) for f in run_dir.glob('*fluxnet*.csv')])
            fullout['date_time'] = pd.to_datetime(fullout['TIMESTAMP_START'], format=r'%Y%m%d%H%M')
            fullout = fullout.drop_duplicates(subset=['date_time'])
            mindate, maxdate = fullout.date_time.min(), fullout.date_time.max()
            new_time = pd.DataFrame(dict(date_time=pd.date_range(mindate, maxdate, freq='30min')))
            fullout = (
                fullout
                .merge(new_time, how='right', on='date_time')
                .sort_values('date_time')
                .set_index('date_time')
            )
            fullout['site'] = site
            all_data.append(fullout)
        except (OSError, ValueError): pass
        if i % 10 == 0: print(i)
        i += 1
    all_data = pd.concat(all_data)
    return all_data