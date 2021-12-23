# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 22:29:38 2021

@author: pcjcl
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 18:02:21 2021

@author: pcjcl
"""


import requests
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
from datetime import datetime
import itertools
import os
from io import StringIO

# =============================================================================
from flask import Flask, request, json, render_template, make_response
import base64
from flask import send_file
from flask_cors import CORS
# 
# #%%
# app = Flask(__name__)
# CORS(app)
# 
# @app.route('/')
# 
# 
# @app.route('/randomflake')
# def random_flake_api():
#     print(f'starting growth at {datetime.now()}')
#     growthcons = conditions()
#             
#     flake,corner1,centre1 = flakegrower(growthcons)
#             
#     animationfile = flake_video(flake,corner1,centre1)
#     
#     print('flake made, about to send')
# 
#     with open(animationfile, 'rb') as f:
#         image_binary = f.read()
#     print('binary read in, about to delete file and send binary')
#     os.remove(animationfile) # delete file after it's been read
#         
#     response = make_response(base64.b64encode(image_binary))
#     response.headers.set('Content-Type', 'image/gif')
#     response.headers.set('Content-Disposition', 'attachment', filename='image.gif')
#     return response
# =============================================================================

#%%
def edge_length4(pts_tup):
    length = 0
    for p in range(1,len(pts_tup)):
        length += ((pts_tup[p][0]-pts_tup[p-1][0])**2+(pts_tup[p][1]-pts_tup[p-1][1])**2)**0.5 # add up the differences between all the points
        
    return length

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def curved_lines(x_vals,y_vals): # make curve lines between all points even if they go back and forth on grid rather than in order
    x_curve = []
    y_curve = []
    for i in range(0,len(x_vals)-1):

        b = (x_vals[i+1]-x_vals[i])/2 + x_vals[i]
        
        a = -1
        c = (b-x_vals[i])**2
        
        if y_vals[i+1] <= y_vals[i] and x_vals[i+1] > x_vals[i]:
            a = 1
            c = -(b-x_vals[i])**2
        elif y_vals[i+1] > y_vals[i] and x_vals[i+1] <= x_vals[i]:
            a = 1
            c = -(b-x_vals[i])**2
        #normalise bulge (height of quadratic part) by the length of the line
        length = ((y_vals[i+1]-y_vals[i])**2+(x_vals[i+1]-x_vals[i])**2)**0.5
        norm = length**0.3/(abs(c)*20)
        if y_vals[i+1]<y_vals[i]:
            norm = -length**0.3/(abs(c)*20)

        m = (y_vals[i+1]-y_vals[i])/(x_vals[i+1]-x_vals[i]) # linear part y = mx * d
        d = y_vals[i] - m*x_vals[i]
        x_=np.linspace(x_vals[i], x_vals[i+1], 50)
        y_= m*x_ + d + (a*(x_-b)**2 + c)*norm


        x_curve.append(np.copy(x_))
        y_curve.append(np.copy(y_))
    x_curve = list(itertools.chain(*x_curve))
    y_curve = list(itertools.chain(*y_curve))
    
    return x_curve, y_curve

def conditions():
    c0 = 2000  # bottom of cloud height, m
    h0 = 5000  # starting height of cloud top of snowflake
    delta_h = 50 # change in height for updating snow flake
    T_h0 = -40 # top of cloud T
    T_c0 = -6 # bottom of cloud T
    T_0 = -5*np.random.default_rng().random() # ground T randomly generated between 0 to -5
    starth = 2500+2500*np.random.default_rng().random()
    
    
    m_p = 1.67262191E-27 # kg
    m_h2o = 18*m_p
    R = 8.3145 # ideal gas constant
    k = 1.38064852E-23 # boltzmanconstant
    
    h = np.linspace(h0, 0, num=round(h0/delta_h)+1)
    T = np.linspace(T_h0,T_c0, num=round((h0-c0)/delta_h))
    T = np.append(T, np.linspace(T_c0,T_0, num=round((c0/delta_h))+1))
    
    startidx = find_nearest(h,starth)
    
    Tnoise = np.random.normal(0,1,len(T)) # (mean, standard deviation, length)
    T = T+Tnoise
    
    VP_H2O = (0.0052*T**3 + 0.7445*T**2 + 35.976*T + 598.87) 
    sinwavperiod = (round((h0-c0)/delta_h))*np.random.default_rng().random()
    RHux = np.linspace(0,(round((h0-c0)/delta_h)),num=(round((h0-c0)/delta_h)))
    RHu = 113 + 4*np.sin(RHux/sinwavperiod)
    RHu_cloudnoise = np.random.normal(0,1.2,round((h0-c0)/delta_h))
    
    RHu = RHu+RHu_cloudnoise
    RHu_0 = np.random.default_rng().random()*100 # random ground relative humidity between 0 and 100 %
    RHu_noise = np.random.normal(0,15,len(h)-len(RHu))
    RHu = np.append(RHu, np.linspace(RHu[-1],RHu_0,num=(len(h)-len(RHu))))
    RHu[round((h0-c0)/delta_h):] += np.random.normal(0,5,len(h)-len(RHu_cloudnoise))
     # adding noise with a std of 10% relative humidity
    
    for r in range(0,len(RHu)): # stop RH going below 0
        if RHu[r] < 0:
            RHu[r] = 0
    #    if RHu[r] > 100:
    #        RHu[r] = 100
    
    over100RHu = RHu - 100*np.ones(len(RHu))
    p_H2O = RHu*VP_H2O/100 # in Pa
    sat_p_H2O = over100RHu/100*VP_H2O
    dens_H2O = m_h2o*p_H2O/(k*(T+273.15))*1000 #T in K remember, * 1000 makes it into grams
    sat_dens_H2O = m_h2o*sat_p_H2O/(k*(T+273.15))*1000 
    
    b=0
    growthrecord = [0,0]
    for i in range(0,len(T)):
        shape_line = -0.0004*T[i]**2 - 0.0164*T[i] + 0.0051
        #print(shape_line)
        if sat_dens_H2O[i] < 0:
            growthweights = [0,0,0,1]
        elif sat_dens_H2O[i] <= shape_line: # formula from fitting 2nd order polynomial to that figure -0.0005x2 - 0.017x
            rem = abs(shape_line-sat_dens_H2O[i]/(shape_line))
            growthweights = [rem,rem/2,rem**3,0]#[abs(rem),(abs(rem**3))/3,abs(rem)**12/3,0]
            #print('below',rem)
        elif sat_dens_H2O[i] > shape_line:
            rem = abs(sat_dens_H2O[i]-(shape_line)/(sat_dens_H2O[i]))
            growthweights = [1-rem/2,1-rem,rem,0] #[abs(rem)**2,abs(rem),(i-b)/(i+1)*abs(rem)**4,0]
            #print('above',rem)
            #print('overline')
        #print(growthweights)
        growthtype = random.choices(['normal','stretch','branch','none'],weights=growthweights,k=1)
        #print(growthtype)
        if growthtype == ['branch']:
            b = i
         #   print(b)
        #print((i-b)/(i+1))
        growthrate = sat_dens_H2O[i]*abs(T[i])**0.5*np.random.default_rng().random() # also a random element to growth. Colder temperatures snow growth a bit higher, and also linear on density of water vapor
        growthrecord = np.vstack([growthrecord, [growthtype,growthrate]])
        
    growthrecord = np.delete(growthrecord, obj=0, axis=0) # deletes that initial row of 0s 
    
    return growthrecord

def flakegrower(growthrecord):
    l = 1
    r = l/2
        
    corner1 = np.array([r, 0])
    centre1a = np.array([0.75*r,3**0.5/4*r])
    
    edge1 = np.array([centre1a,corner1])
    points_tuple = tuple(map(tuple, edge1))
    
    # rotation matrices for branching at +- 60 degrees
    R1 = np.array(((np.cos(np.radians(60)), -np.sin(np.radians(60))),
                   (np.sin(np.radians(60)), np.cos(np.radians(60))))) 
    R2 = np.array(((np.cos(np.radians(-60)), -np.sin(np.radians(-60))),
                   (np.sin(np.radians(-60)), np.cos(np.radians(-60)))))
    branch1v = np.matmul(corner1/np.linalg.norm(corner1),R2)
    branch2v = np.matmul(corner1/np.linalg.norm(corner1),R1)
    branchvs = np.array((1,1))
    branch_origin = np.array((1,1))
    Refx = np.array(((1,0),(0,-1)))  #reflect in the x axis
    stretch = np.zeros([100,2])
    
    all_flakes = [np.copy(edge1)] # first one is the hexagon
    
    br = 0
    for g in range(0,len(growthrecord[:,])):
        edge_length = (2*edge_length4(points_tuple))**0.5
        area = growthrecord[g,1]/6 #  /6 because it's doing it on 6 sides
        #print(edge_length,area)
        if growthrecord[g,0] == ['none']:
            blob = 1
        if growthrecord[g,0] == ['normal']: 
            if br >= 0: # realised this one (originally br =0) unneccesary just need branch conditions later but i didn't want to delete all the tabbings
                for e in edge1:
                    #print(e/np.linalg.norm(e))
                    if np.allclose(e/np.linalg.norm(e),corner1/np.linalg.norm(corner1),rtol=1e-05) == True: # allclose works better than array_equal because has tolerance incase of stupid float maths
                        e += 10.1*area/edge_length*corner1#/np.linalg.norm(corner1)
                        #print('c')
                    elif np.allclose(e/np.linalg.norm(e),centre1a/np.linalg.norm(centre1a),rtol=1e-05) == True:
                        if br < 1:
                            e += 1*area/edge_length*centre1a
                        if br >= 1:
                            e += 0.1*area/edge_length*centre1a # slow down centre faces of starting hexagon growth after branched
                   
                    if br > 0: # add branch points growth
                            for bra in range(0,len(branchvs)):
                                #if bra % 2 == 0: # adding random aspect to branch growth, didn't work so well
                                    #brgrowth = br*2*np.random.default_rng().random()*area/edge_length # update the growth rate only on the even numbers so that the +- pairs grow equally
                                brgrowth = br*area/edge_length # update the growth rate only on the even numbers so that the +- pairs grow equally
                                    
                                if np.allclose((e-branch_origin[bra])/np.linalg.norm(e-branch_origin[bra]),
                                               branchvs[bra]/np.linalg.norm(branchvs[bra]),rtol=1e-05) == True:                               
                                    if abs(e[1]+brgrowth*branchvs[bra][1]) < (e[0]+brgrowth**branchvs[bra][0])*0.42: # stop the branches from crossing over. (I thought it should be < x*0.577 / tan30 but seems to still cross)
                                        e += brgrowth*branchvs[bra]
                                    else:
                                        pass
                    #print(g,e,br)
                                    
            if br > 0:
                    for i in range(0,len(edge1)):
                        if np.allclose(edge1[i],edget[i],rtol=1e-07) == True:
                            #print(edge1[i],edget[i],g)
                            if edge1[i][1] > 0:
                                edge1[i] += 1.2*area/edge_length*(np.array((0,1)))
                                
                                #print(edge1)
                            elif edge1[i][1] < 0:
                                edge1[i] += 1.2*area/edge_length*(np.array((0,-1)))
                                
        if growthrecord[g,0] == ['stretch']: #just make it so that only outer corners grow
            for e in edge1:
                if np.allclose(e/np.linalg.norm(e),corner1/np.linalg.norm(corner1),rtol=1e-05) == True:     
                     e += 0.5*area*corner1 # stretch the corner points
                     
                     
        if growthrecord[g,0] == ['branch']:
            for e in range(0,len(edge1)):
                if np.allclose(edge1[e]/np.linalg.norm(edge1[e]),corner1/np.linalg.norm(corner1),rtol=1e-05) == True: #branching off centre line only. can add elifs for branches branching later
                    #print(e)
                    brwidth = 1/edge1[e][0] # normalise branch width by the value (length) of e (stops the branches getting out of control chunk as the snowflake grows)
                    if brwidth > 0.15:
                        brwidth = 0.14
                    #print(brwidth)
                    branch_origin = np.vstack([branch_origin, [(0.85+brwidth/2)*(edge1[e]-edge1[0])+edge1[0]],[np.matmul((0.85+brwidth/2)*(edge1[e]-edge1[0])+edge1[0],Refx)]])

                    edget = np.copy(edge1)
                    #print(edget)
                    if br <1:
                        edget = np.delete(edget, e,axis=0) # remove the original corner for now so it can bet put in the middle of the branches axis = 0 makes it keep its shape
                        edget = np.vstack([edget,[0.85*(edge1[e]-edge1[0])+edge1[0]],
                                           [((0.85+brwidth/2)*(edge1[e]-edge1[0])+edge1[0]+branch1v*area/edge_length)],
                                           [(0.85+brwidth)*(edge1[e]-edge1[0])+edge1[0]],
                                           [edge1[e]],[np.matmul((0.85+brwidth)*(edge1[e]-edge1[0])+edge1[0],Refx)],
                                           [(np.matmul((0.85+brwidth/2)*(edge1[e]-edge1[0])+edge1[0],Refx)+branch2v*area/edge_length)],
                                           [np.matmul(0.85*(edge1[e]-edge1[0])+edge1[0],Refx)]])
                        edge1 = np.copy(edget)
                        #print(edget)
                        
                    elif br >= 1:
                        edget = np.delete(edget, np.s_[e:len(edge1)],axis=0) # delete outer point corner point on x axis and everything after it
                        #print(edget)
                        # put in the new branch coordinates as before with the outer point in the centre
                        edget = np.vstack([edget,[0.85*(edge1[e]-edge1[0])+edge1[0]],
                                           [((0.85+brwidth/2)*(edge1[e]-edge1[0])+edge1[0]+branch1v*area/edge_length)],
                                           [(0.85+brwidth)*(edge1[e]-edge1[0])+edge1[0]],[edge1[e]],
                                           [np.matmul((0.85+brwidth)*(edge1[e]-edge1[0])+edge1[0],Refx)],
                                           [(np.matmul((0.85+brwidth/2)*(edge1[e]-edge1[0])+edge1[0],Refx)+branch2v*area/edge_length)],
                                           [np.matmul(0.85*(edge1[e]-edge1[0])+edge1[0],Refx)]])
                        # put the rest of the points you deleted back in
                        for k in range(e+1,len(edge1)):
                            edget = np.vstack([edget, [edge1[k]]])
                        edge1 = np.copy(edget)
                    branchvs = np.vstack([branchvs,[branch1v],[branch2v]]) # add the growth vectors to vs    
                    break
                    
            br += 1
                 
        points_tuple = tuple(map(tuple, edge1)) # overwrite points tuple
        all_flakes.append(np.copy(edge1))    

    all_flakes_rot = []

    for f in all_flakes:
        Rottemp = f
        for n in range(1,6):
            theta = np.radians(60*n)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))      
            for e in f:
                Rottemp = np.vstack([Rottemp,[np.matmul(e, R)]])
    
        all_flakes_rot.append(np.copy(Rottemp))
    
    return all_flakes_rot,corner1,centre1a

def snowflake_animation(g,all_flakes_rot,snowflake):#,points_tuple,br,growthrecord):
        x_values = []
        y_values = []
        for s in all_flakes_rot[g]:
            x_values = np.append(x_values,s[0])
            y_values = np.append(y_values,s[1])
        x_values = np.append(x_values,all_flakes_rot[g][0][0]) # complete the pattern
        y_values = np.append(y_values,all_flakes_rot[g][0][1])
        x_c, y_c = curved_lines(x_values, y_values)
        snowflake.set_xdata(x_c)
        snowflake.set_ydata(y_c)
    
        return snowflake,#points_tuple,br,edge1,growthrecord,branch_origin,branchvs,edget,

def flake_video(all_flakes_rot,corner1,centre1a):
    x_values = []
    y_values = []
    Rottemp = np.array([centre1a,corner1])
    for n in range(1,6):
            theta = np.radians(60*n)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))       
    for e in range(0,len(all_flakes_rot[0])):
        x_values = np.append(x_values,all_flakes_rot[0][e][0])
        y_values = np.append(y_values,all_flakes_rot[0][e][1])
    x_values = np.append(x_values,all_flakes_rot[0][0][0]) # complete the pattern
    y_values = np.append(y_values,all_flakes_rot[0][0][1])
    x_c, y_c = curved_lines(x_values, y_values)
    
    fig, ax = plt.subplots()

    ax.set_xlim(-1.2*np.max(all_flakes_rot[-1]),1.2*np.max(all_flakes_rot[-1]))
    ax.set_ylim(-1.2*np.max(all_flakes_rot[-1]),1.2*np.max(all_flakes_rot[-1]))    
    ax.set_facecolor('k')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    snowflake, = ax.plot(x_c, y_c,linewidth=1.5)
    #snowflake, = ax.fill_between(x_values, y_values,color=[(0.8,0.9,1)])

  
    flakeanimation = animation.FuncAnimation(fig, func=snowflake_animation,
                                             frames=np.arange(0,len(all_flakes_rot),1),
                                             interval=100, fargs=(all_flakes_rot,snowflake,)) #interval in ms, frames is the i values
    #plt.show
    plt.close() 
        
    #to hard save file:        
    dt_now = str(datetime.now())
    dt_now = dt_now[:19] # current time up to the seconds
    dt = dt_now.replace(':', '-') # formatting so it can be used as a file name
    animationfilename = 'snowflakeanimation_'+dt+'.gif'


    #animationfilename.write(str(flakeanimation))
    flakeanimation.save((animationfilename), writer='imagemagick', fps=10) # could not work out how to save animation.funcanimation object in the binary so saving real file for now and deleting after sending on server
    print('flake animation successfully created')
    return animationfilename


