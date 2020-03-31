# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:14:11 2020

@author: Kanchana
"""
import progressbar

def read_file(filename):
    data = []
    count = 0
    with open(filename,'r') as f:
        for line in f:
            count = count + 1
            if(count > 6):
                terms = line.strip().split(',')
                data.append(np.asarray(list(map(float,terms[:2]))))
    return data

def write_file(filename,data):
    f = open(filename,'w',encoding = 'utf-8-sig')
    lines = []
    for d in data:
        l = ','.join(list(map(str,d))) 
        lines.append((l+"\n"))
    f.writelines(lines)
    f.close()


def getDist(Q,trajectories,method = 'paper1'):
    metric = DistanceMetric(Q)
    N = len(trajectories)
    D = [[0 for i in range(N)] for j in range(N)]
    print("The number of trajectories: %d"%N)
    with progressbar.ProgressBar(max_value=N) as bar:
        for i in range(N):
            #print("Current trajectory id: %d"%i)
            bar.update(100.0*i/N)
            for j in range(i,N):
                if(i!=j):
                    if(method=='paper1'):
                        d = metric.calc_trajectorydst_opt(Q,trajectories[i],trajectories[j])[1]
                    if(method=='paper2'):
                        d = metric.calc_trajectorydst_opt(Q,trajectories[i],trajectories[j])[0]
                    if(method=='euclid'):
                        d = metric.calc_euclideandst(trajectories[i],trajectories[j])
                    if(d == float('inf')):
                        print("Traj: %d, %d"%(i,j))
                    D[i][j] = d
                    D[j][i] = d
                else:
                    D[i][j] = 0
    return D

def convertToLine(data):
    lines = []
    for i in range(len(data)-1):
        if(not(data[0][0]==data[1][0] and data[0][1]==data[1][1])):
            l = Line(data[0],data[1])
            lines.append(l)
    return lines

def main():
    random.seed(0)
    files = ['000','001']
    traj_lst = {'000':[] , '001':[]}
    for name in files:
        dirpath = "ExtractedData/"+name+"/CSV/*.csv"
        dirlst = glob.glob(dirpath)
        trajectories = []
        lnths = []
        for f in dirlst:
            data = read_file(f)
            lines = convertToLine(data)
            if(len(lines)>100):
                t = Trajectory(lines)
                trajectories.append(t)
                lnths.append(len(lines))
        lnths = np.asarray(lnths)
        idx = lnths.argsort()[::-1]
        traj = []
        for i in idx:
            traj.append(trajectories[i])
        traj_lst[name] = traj
    Q = []
    N = 10
    while(len(Q)<N):
        x = random.randrange(437030,467040)
        y = random.randrange(4416500,4436700)
        p = [x,y]
        if(not(p in Q)):
            Q.append(p)
    for i in range(len(Q)):
        Q[i] = np.asarray(Q[i])
    sze = min(len(traj_lst['000']),len(traj_lst['001']))
    err_ = []
    T = 1
    #idx = random.sample(list(range(sze)),10)  
    idx = list(range(sze))  
    all_traj = []
    for i in idx:
        all_traj.append(traj_lst['000'][i])
    for i in idx:
        all_traj.append(traj_lst['001'][i])
    #all_traj = traj_lst['000'][idx] + traj_lst['001'][idx]
    #printDist(Q,all_traj)
    print('Working out the distances')
    D = getDist(Q,all_traj,method = 'paper1')
    print('Writing to the file')
    write_file('dist2.csv',D)
        
if __name__=="__main__":main()