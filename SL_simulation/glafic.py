import os 
class glafic:
    def __init__(self):
        self.base = ''
        self.prefix = 'out'

    def init(self,
            omega = 0.3,
            lamb = 0.7,
            weos = -1,
            h = 0.7,
            prefix = 'out',
            xmin = -5,
            ymin = -5,
            xmax = 5,
            ymax = 5,
            pix_ext = 0.1,
            pix_poi = 0.1,
            maxlev = 5,
             ):
        '''
        
        Parameters
        ----------
        omega : float, optional
            standard cosmological parameter
        lamb : float, optional
            standard cosmological parameter
        weos : float, optional
            standard cosmological parameter
        h : float, optional
            standard cosmological parameter
        prefix : str, optional
            the prefix of the output files
        xmin,ymin,xmax,ymax : float, optional  
            These parameters specify the rectangular region of the lens plane in which lens equation is solved (in units of arcsecond)
        pix_ext : float, optional
            the pixel size for the extended source
        pix_poi : float, optional
            the largest pixel (grid) size for point sources, in units of arcsecond
        maxlev : int, optional
            the maximum recursion level of adaptive meshing
        '''
        self.base += '## setting primary parameters \n'
        self.base += f'omege {omega} \n'
        self.base += f'lambda {lamb} \n'
        self.base += f'weos {weos} \n'
        self.base += f'h {h} \n'
        self.base += f'prefix {prefix} \n'
        self.prefix = prefix
        self.base += f'xmin {xmin} \n'
        self.base += f'ymin {ymin} \n'
        self.base += f'xmax {xmax} \n'
        self.base += f'ymax {ymax} \n'
        self.base += f'pix_ext {pix_ext} \n'
        self.base += f'pix_poi {pix_poi} \n'
        self.base += f'maxlev {maxlev} \n'
        self.base += '\n'

    def set_secondary(self,input):
        '''
        
        Parameters
        ----------
        input : str
            the input string
        '''
        self.base += input + '\n'


    def startup_setnum(self,num_lens, num_extsrc, num_pntsrc):
        '''
        :param num_lens: the number of lenses
        :param num_extsrc: the number of extended sources
        :param num_pntsrc: the number of point sources
        '''
        out_str = f'startup {num_lens} {num_extsrc} {num_pntsrc} \n'
        self.base += out_str


    def set_lens_gnfw(self, z, M_lens, x,y,e,pa,p7,gamma):
        '''
        
        Parameters
        ----------
        z : float
            redshift
        M_lens : float
            the mass scale of lens
        x : float(arcsec)
            the x-position of the lens
        y : float(arcsec)
            the y-position of the lens
        e : float
            the ellipticity of the lens
        pa : float
            the position angle of the lens
        p7 : float
            the concentration parameter
        gamma : float   
            the inner slope of the density profile
        '''
        out_str = f'{lens_id} gnfw {z} {M_lens} {x} {y} {e} {pa} {p7} {gamma} \n'
        self.base += out_str

    def set_lens_sers(self, z, M_lens, x,y,e,pa,re, n):
        """

        Parameters
        ----------
        z : float
            redshift
        M_lens : float
            the mass scale of lens
        x : float(arcsec)
            the x-position of the lens
        y : float(arcsec)
            the y-position of the lens
        e : float
            the ellipticity of the lens
        pa : float
            the position angle of the lens
        re : float  
            effective radius
        n : float
            the Sersic index
        """
        out_str = f'lens sers {z} {M_lens} {x} {y} {e} {pa} {re} {n} \n'
        self.base += out_str

    def set_point(self, z, x,y):
        """

        Parameters
        ----------
        z : float
            redshift
        x : float(arcsec)
            x_position
        y : float(arcsec)
            y_position
        """        
        out_str = f'point {z} {x} {y} \n'
        self.base += out_str

    def model_init(self):
        self.base += 'end_startup \n'
        self.base += '\n' 
        self.base += 'start_command \n'

    def findimg(self):
        pre = self.base
        out = pre + 'findimg \n' + 'quit'
        with open(f'{self.prefix}.input','w') as f:
            f.write(out)
        os.system(f'/Users/liurongfu/softwares/glafic2/glafic {self.prefix}.input')
        

        



    
    
     